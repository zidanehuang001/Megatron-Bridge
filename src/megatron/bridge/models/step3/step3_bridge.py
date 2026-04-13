# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
import torch.nn as nn
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


class _Step3FC1ExpertMapping(GatedMLPMapping):
    """Maps two separate 3D stacked gate/up expert tensors to Megatron per-expert FC1.

    Step-3.5-Flash stores MoE expert projections as batched 3D tensors via MoELinear:
        gate_proj: [num_experts, intermediate_size, hidden_size]
        up_proj:   [num_experts, intermediate_size, hidden_size]

    Megatron expects per-expert fused FC1 weights: [gate_shard; up_shard]

    HF → Megatron (import): Extracts per-expert slices from 3D tensors and
        concatenates gate + up for each expert.

    Megatron → HF (export): Not yet supported. The 3D stacked tensors require a
        two-key grouped export that is not handled by the current is_grouped_export
        protocol. A future implementation can split gate and up from FC1 and stack
        them into separate 3D tensors.
    """

    is_grouped_export = True

    def __init__(self, megatron_param: str, gate: str, up: str):
        super().__init__(megatron_param, gate, up)
        self.allow_hf_name_mismatch = True

    @property
    def group_key(self) -> str:
        # Use gate param as group key to enable HF import caching of the 3D tensor.
        return self.hf_param["gate"]

    def hf_to_megatron(
        self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module
    ) -> torch.Tensor:
        """Extract per-expert gate and up slices from 3D stacked tensors, then fuse."""
        expert_idx = extract_expert_number_from_param(self.megatron_param)
        gate = hf_weights["gate"][expert_idx]  # [intermediate_size, hidden_size]
        up = hf_weights["up"][expert_idx]  # [intermediate_size, hidden_size]
        # Delegate to GatedMLPMapping which handles TP scatter of the fused [gate; up] weight.
        return super().hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Export not yet supported for 3D stacked expert weights (returns empty)."""
        return {}

    def resolve(self, captures):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["gate"],
            resolved_hf_param["up"],
        )


@MegatronModelBridge.register_bridge(
    source="Step3p5ForCausalLM",
    target=GPTModel,
    model_type="step3p5",
)
class Step3Bridge(MegatronModelBridge):
    """Megatron Bridge for Step-3.5-Flash Causal LM.

    Step-3.5-Flash is a 196.81B sparse MoE model (~11B active parameters) with a
    custom ``step3p5`` architecture featuring:

    - 45 transformer layers (layers 0–2 dense MLP, layers 3–44 MoE)
    - 288 routed experts + 1 shared expert per MoE layer (top-8 sigmoid routing)
    - Expert bias correction for load balancing
    - QK RMSNorm (per-head normalization of queries and keys)
    - Zero-centered RMSNorm throughout
    - GQA: 64 query heads / 8 KV heads, head_dim=128
    - RoPE with llama3-style scaling (θ=5,000,000)

    Architecture notes:
    - MoE expert weights are stored as batched 3D tensors (MoELinear).
      FC1 (gate+up) import is handled by ``_Step3FC1ExpertMapping``; export is not
      yet supported and must be done separately if needed.
    - Head-wise attention gating (g_proj) is not mapped in this bridge.
      Megatron will randomly initialize this parameter; it will be learned during
      training or can be added in a future revision.
    - Sliding window attention (alternating full / SWA) is not implemented.
      Full attention is used for all layers.
    - Per-layer RoPE theta (rope_theta array) is simplified to a single scalar
      (5,000,000). Advanced llama3-style RoPE scaling for 256K context needs
      additional configuration.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "stepfun-ai/Step-3.5-Flash", trust_remote_code=True
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace Step-3.5-Flash config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # ── Core transformer settings ──────────────────────────────────────────
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16
        # Zero-centered RMSNorm: weight init=0, forward = x * (1 + w) / rms(x)
        provider.layernorm_zero_centered_gamma = True
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # ── Attention ────────────────────────────────────────────────────────
        # Step-3.5-Flash uses num_attention_groups instead of num_key_value_heads
        num_kv_heads = getattr(hf_config, "num_attention_groups", None) or getattr(
            hf_config, "num_key_value_heads", None
        )
        if num_kv_heads is not None:
            provider.num_query_groups = num_kv_heads  # 8

        # Per-layer RoPE theta: HF stores a 48-element array; slice to num_hidden_layers.
        # Falls back to scalar 5,000,000 if the field is absent or scalar.
        rope_theta = getattr(hf_config, "rope_theta", None)
        if isinstance(rope_theta, list):
            provider.rotary_base_per_layer = rope_theta[:hf_config.num_hidden_layers]
        else:
            provider.rotary_base = float(rope_theta) if rope_theta else 5_000_000.0

        # Per-head scalar output gate (g_proj)
        provider.attention_per_head_gate = getattr(hf_config, "use_head_wise_attn_gate", False)

        # ── MoE layer distribution ───────────────────────────────────────────
        # Layers 0–2 are dense MLP; layers 3–44 are MoE.
        # moe_layers_enum is a comma-separated string: "3,4,5,...,44"
        moe_layers_str = getattr(hf_config, "moe_layers_enum", None)
        if moe_layers_str:
            moe_layer_indices = {int(x) for x in moe_layers_str.split(",")}
            first_moe_layer = min(moe_layer_indices)
        else:
            first_moe_layer = 3  # default for Step-3.5-Flash
        num_layers = hf_config.num_hidden_layers  # 45
        provider.moe_layer_freq = [0] * first_moe_layer + [1] * (num_layers - first_moe_layer)

        # ── MoE hyperparameters ───────────────────────────────────────────────
        # HF uses non-standard field names that don't match CONFIG_MAPPING entries.
        provider.num_moe_experts = getattr(hf_config, "moe_num_experts", 288)
        provider.moe_router_topk = getattr(hf_config, "moe_top_k", 8)

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        # Sigmoid routing with expert bias correction
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = getattr(hf_config, "use_moe_router_bias", True)
        provider.moe_router_dtype = "fp32"

        # Shared expert (always active, no output gate unlike Qwen3-Next)
        provider.moe_shared_expert_gate = False
        provider.moe_shared_expert_intermediate_size = getattr(hf_config, "share_expert_dim", 1280)

        # ── Sliding Window Attention ──────────────────────────────────────────
        # layer_types is a 45-element list; "sliding_attention" maps to 1 (SWA),
        # all other values map to 0 (full attention).
        # Mcore's window_attn_skip_freq already supports List[int].
        layer_types = getattr(hf_config, "layer_types", None)
        sliding_window = getattr(hf_config, "sliding_window", None)
        if layer_types and sliding_window:
            provider.window_size = (sliding_window, 0)
            provider.window_attn_skip_freq = [
                1 if lt == "sliding_attention" else 0 for lt in layer_types
            ]

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return parameter mappings between Megatron and HuggingFace formats."""

        # ── 1:1 name mappings ─────────────────────────────────────────────────
        param_mappings = {
            # Global embeddings and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Per-layer attention norms (all 45 layers)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # Attention output projection (all 45 layers)
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Per-head scalar gate (g_proj): Linear(hidden_size → num_heads)
            "decoder.layers.*.self_attention.linear_gate.weight": "model.layers.*.self_attn.g_proj.weight",
            # Pre-MLP layernorm for MoE layers (3–44)
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Pre-MLP layernorm fused into FC1 for dense layers (0–2)
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MoE router (layers 3–44)
            "decoder.layers.*.mlp.router.weight": "model.layers.*.moe.gate.weight",
            # MoE router expert bias correction
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.moe.router_bias",
            # Dense MLP output projection (layers 0–2)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Shared expert output projection (layers 3–44; no gate unlike Qwen3-Next)
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.share_expert.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # ── QKV concatenation (all 45 layers) ────────────────────────────────
        # q: [num_heads * head_dim, hidden]  k/v: [num_kv_heads * head_dim, hidden]
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # ── Dense MLP FC1 (layers 0–2): standard gate+up concat ───────────────
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            )
        )

        # ── MoE routed expert weights (layers 3–44) ───────────────────────────
        # Expert weights are stored as batched 3D tensors (MoELinear):
        #   gate_proj / up_proj: [num_experts, intermediate_size, hidden_size]
        #   down_proj:           [num_experts, hidden_size, intermediate_size]
        #
        # FC1 = gate + up (custom mapping handles the 3D batched format)
        mapping_list.append(
            _Step3FC1ExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.moe.gate_proj.weight",
                up="model.layers.*.moe.up_proj.weight",
            )
        )
        # FC2 = down_proj (FusedExpertMapping handles [N, out, in] → per-expert)
        mapping_list.append(
            FusedExpertMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                hf_param="model.layers.*.moe.down_proj.weight",
            )
        )

        # ── Shared expert FC1 / FC2 (layers 3–44) ────────────────────────────
        # Shared expert uses standard Step3p5MLP (per-module, not MoELinear)
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.share_expert.gate_proj.weight",
                    up="model.layers.*.share_expert.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
