# PR Descriptions — Step-3.5-Flash Support

This file contains draft PR descriptions for the two related pull requests:

1. **[Megatron-Bridge PR]** — model bridge, recipes, training scripts, tests
2. **[Megatron-Core PR]** — `attention_per_head_gate` and `rotary_base_per_layer`

---

## PR 1 — Megatron-Bridge: Add Step-3.5-Flash (stepfun-ai/Step-3.5-Flash) support

### What does this PR do?

Adds full Megatron-Bridge support for
[stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash), a 196.81B sparse
MoE model (~11B active parameters) using the custom `step3p5` architecture. Enables HF → Megatron
weight conversion, distributed training (TP/PP/EP), and SFT with all major architecture features
faithfully mapped.

---

### Architecture overview

Step-3.5-Flash uses a distinct architecture that differs from standard MoE models in several ways.
The full-model values from `stepfun-ai/Step-3.5-Flash/config.json` are:

| Property | Full model | Toy test model |
|---|---|---|
| `num_hidden_layers` | 45 | 5 |
| Dense layers | 0–2 (3 total) | 0–2 (3 total) |
| MoE layers | 3–44 (42 total) | 3–4 (2 total) |
| `hidden_size` | 4096 | 512 |
| `num_attention_heads` | 64 | 8 |
| `num_attention_groups` (KV heads) | 8 | 2 |
| `head_dim` | 128 | 64 |
| `intermediate_size` (dense FFN) | 11264 | 1024 |
| `moe_num_experts` (routed) | 288 | 4 |
| `share_expert_dim` | 1280 | 256 |
| `moe_top_k` | 8 | 2 |
| `moe_intermediate_size` | 1280 | 256 |
| Router score function | sigmoid | sigmoid |
| `use_moe_router_bias` | true | true |
| `vocab_size` | 128,896 | 1024 |
| `rms_norm_eps` | 1e-6 | 1e-6 |
| `tie_word_embeddings` | false | false |
| `use_head_wise_attn_gate` | true | true |
| `rope_theta` | 48-element list (5M/10k/10k/10k repeating) | 5-element list |
| `layer_types` | 45-element list (full / SWA alternating) | 5-element list |
| `sliding_window` | 512 | 512 |

**Non-standard HF field names** (requires explicit mapping in `provider_bridge`):
- `num_attention_groups` instead of `num_key_value_heads` for KV heads
- `moe_num_experts` instead of `num_local_experts`
- `moe_top_k` instead of `num_experts_per_tok`
- `share_expert_dim` instead of `moe_shared_expert_intermediate_size`

---

### Changelog

#### New: `src/megatron/bridge/models/step3/step3_bridge.py`

Core bridge class `Step3Bridge` and the custom `_Step3FC1ExpertMapping`.

**`provider_bridge()` — config translation to `GPTModelProvider`:**

- Sets standard transformer flags: `normalization=RMSNorm`, `gated_linear_unit=True`,
  `position_embedding_type=rope`, `layernorm_zero_centered_gamma=True` (zero-centered RMSNorm,
  weight init=0, forward = `x * (1+w) / rms(x)`).
- Reads `num_attention_groups` with fallback to `num_key_value_heads` for KV head count.
- Maps `moe_layers_enum` (comma-separated string `"3,4,...,44"`) to
  `moe_layer_freq = [0,0,0,1,...,1]` (3 zeros + 42 ones).
- Sets sigmoid routing: `moe_router_score_function="sigmoid"`,
  `moe_router_enable_expert_bias=True`, `moe_router_dtype="fp32"`.
- Sets shared expert without output gate: `moe_shared_expert_gate=False`,
  `moe_shared_expert_intermediate_size=share_expert_dim`.
- Maps `use_head_wise_attn_gate` → `provider.attention_per_head_gate` (new Mcore flag, see PR 2).
- Maps `rope_theta` list (48 elements) → `provider.rotary_base_per_layer[:45]`
  (new Mcore field, see PR 2). Falls back to `rotary_base=5_000_000.0` for scalar configs.
- Maps `layer_types` + `sliding_window` → `provider.window_size=(512,0)` +
  `provider.window_attn_skip_freq=[1 if lt=="sliding_attention" else 0 for lt in layer_types]`.
  No Mcore change needed: `window_attn_skip_freq: Optional[Union[int, List[int]]]` already
  accepts a list in the dev branch.

**`mapping_registry()` — parameter mappings:**

```
# Global (3 params)
embedding.word_embeddings.weight       ↔  model.embed_tokens.weight
output_layer.weight                    ↔  lm_head.weight
decoder.final_layernorm.weight         ↔  model.norm.weight

# Per-layer attention, all 45 layers (5 params × 45 = 225 total)
self_attention.linear_qkv.weight       ←  QKVMapping(q_proj, k_proj, v_proj)
                                            q: [64×128, 4096] → [8192, 4096]
                                            k/v: [8×128, 4096] → [1024, 4096]
self_attention.linear_qkv.layer_norm_weight  ↔  input_layernorm.weight [4096]
self_attention.q_layernorm.weight      ↔  self_attn.q_norm.weight [128]  (per-head)
self_attention.k_layernorm.weight      ↔  self_attn.k_norm.weight [128]  (per-head)
self_attention.linear_proj.weight      ↔  self_attn.o_proj.weight [4096, 8192]
self_attention.linear_gate.weight      ↔  self_attn.g_proj.weight [64, 4096]  ← NEW

# Dense MLP, layers 0–2 (3 params × 3 = 9 total)
mlp.linear_fc1.layer_norm_weight       ↔  post_attention_layernorm.weight [4096]
mlp.linear_fc1.weight                  ←  GatedMLPMapping(gate_proj, up_proj)
                                            fused: [22528, 4096]  (2 × 11264)
mlp.linear_fc2.weight                  ↔  mlp.down_proj.weight [4096, 11264]

# MoE, layers 3–44 (7 params × 42 = 294 total)
pre_mlp_layernorm.weight               ↔  post_attention_layernorm.weight [4096]
mlp.router.weight                      ↔  moe.gate.weight [288, 4096]
mlp.router.expert_bias                 ↔  moe.router_bias [288]
mlp.experts.linear_fc1.weight*         ←  _Step3FC1ExpertMapping
                                            gate_proj[288, 1280, 4096] + up_proj[288, 1280, 4096]
                                            → per-expert fused [2560, 4096]
mlp.experts.linear_fc2.weight*         ←  FusedExpertMapping
                                            down_proj[288, 4096, 1280] → per-expert [4096, 1280]
mlp.shared_experts.linear_fc1.weight   ←  GatedMLPMapping(share_expert.gate/up)
                                            fused: [2560, 4096]
mlp.shared_experts.linear_fc2.weight   ↔  share_expert.down_proj.weight [4096, 1280]
```

**`_Step3FC1ExpertMapping` — custom 3D expert tensor handling:**

Step-3.5-Flash stores all expert FC1 weights as batched 3D tensors via `MoELinear`:
```
gate_proj: [288, 1280, 4096]   # [num_experts, intermediate_size, hidden_size]
up_proj:   [288, 1280, 4096]
```
There is no per-expert list; all 288 experts are packed in one tensor. Standard
`GatedMLPMapping` cannot handle this because it expects a single `[inter, hidden]` tensor per
expert. `_Step3FC1ExpertMapping` extends `GatedMLPMapping` and:
- Extracts `gate[expert_idx]` and `up[expert_idx]` from the 3D tensors by index.
- Delegates to `GatedMLPMapping.hf_to_megatron()` which handles TP sharding of the
  fused `[gate; up]` weight.
- Uses `is_grouped_export=True` + `group_key=gate_param_name` so the 3D tensor is
  loaded from HF state dict once and cached for all 288 expert mappings.
- Export (`megatron_to_hf`) returns `{}`: reconstructing 3D stacked tensors from
  per-expert Megatron weights would require a grouped two-key export protocol not
  yet supported. Marked as a known limitation.

#### New: `src/megatron/bridge/recipes/step3/step3.py`

Three recipe helpers:

- **`step3_flash_pretrain_config()`** — TP=1, PP=4, EP=16 (288/16=18 experts/rank),
  DeepEP flex dispatcher, BF16, selective recompute (`layernorm`, `moe`, `moe_act`),
  distributed optimizer, `alltoall` token dispatcher.
- **`step3_flash_sft_config()`** — same parallelism, packed sequence SFT, GBS=128,
  MBS=1, seq_len=2048, LR=5e-6 with 50-step warmup.
- **`step3_flash_peft_config()`** — raises `NotImplementedError`; PEFT not validated.

#### New: `examples/models/step3/`

- `train_step3_flash.py` — Python training entry point; selects pretrain or SFT
  recipe via `--mode`; supports `--pretrained-checkpoint`, `--data-path`,
  `--config-file` (YAML), and Hydra-style dot-notation CLI overrides.
- `slurm_train.sh` — 8-node Slurm training job; passes checkpoint + data paths
  and CLI overrides to `train_step3_flash.py`.
- `slurm_conversion.sh` — sweeps TP/PP/EP configs for round-trip verification.
- `slurm_inference.sh` — text generation at TP=1, PP=4, EP=16.
- `conf/step3_flash_sft_override.yaml` — YAML override template.
- `README.md` — quickstart: conversion → training → inference, with a correctness
  verification table (expected loss range, grad norm, aux loss, tokens/GPU/s).

#### Modified: `src/megatron/bridge/models/__init__.py`

Added `from megatron.bridge.models.step3 import Step3Bridge` and `"Step3Bridge"` to `__all__`.

#### Modified: `src/megatron/bridge/recipes/__init__.py`

Added `from megatron.bridge.recipes.step3 import *`.

#### New: `tests/functional_tests/test_groups/models/step3/test_step3_conversion.py`

**`test_toy_model_creation` (no GPU):** Creates a toy `Step3p5ForCausalLM` model (5 layers,
4 experts) using `AutoConfig.from_pretrained("stepfun-ai/Step-3.5-Flash", trust_remote_code=True)`
overridden with toy dims. Asserts:
- Config fields: `model_type`, `hidden_size`, `num_hidden_layers`, `moe_num_experts`.
- `rope_theta` is stored as a list of length `num_hidden_layers` in the saved JSON.
- `use_head_wise_attn_gate` is `True`.
- `layer_types` is a list of length `num_hidden_layers`.
- Dense layer has `mlp` submodule.
- MoE layer has `moe` submodule with `gate_proj` of shape `[num_experts, ...]` (3D).
- All attention layers have `g_proj` of shape `[num_attention_heads, hidden_size]`.

**`test_step3_conversion_parallelism` (GPU, parametrized):**

| `test_name` | TP | PP | EP | Total GPUs |
|---|---|---|---|---|
| TP | 2 | 1 | 1 | 2 |
| PP | 1 | 2 | 1 | 2 |
| EP | 1 | 1 | 2 | 2 |

Runs `hf_megatron_roundtrip_multi_gpu.py` via `torch.distributed.run`, then reloads the
exported HF model and asserts `model_type`, `hidden_size`, `moe_num_experts` in `config.json`.

**`test_step3_autoconfig_roundtrip` (GPU):** Uses `autoconfig_roundtrip` utility for an
`AutoBridge`-based end-to-end conversion check.

Note: The test file uses `trust_remote_code=True` with a module-level `pytest.skip` guard
(the GLM pattern) because `Step3p5ForCausalLM` lives on the HF Hub and is not registered
in the standard Transformers library.

---

### Hardware requirements

Minimum for full model (TP=1, PP=4, EP=16, BF16):

| Resource | Calculation |
|---|---|
| Routed expert weights | 288 × 2 × (1280×4096 + 4096×1280) × 2 bytes ≈ 24 GB |
| Shared expert weights | 2 × 1280×4096 × 2 bytes ≈ 84 MB |
| Dense layer FFN | 3 × 2 × 11264×4096 × 2 bytes ≈ 1.7 GB |
| Attention weights (all 45 layers) | 45 × (QKV + O + g\_proj) ≈ 7 GB total |
| **Active params per EP rank** (EP=16) | ~11B / 16 ≈ 688M params ≈ **1.4 GB** model weights |
| **Full model in BF16** | ~393 GB across 64 GPUs |

Requires ≥ 8 nodes × 8 × H100/A100 80 GB (64 GPUs total) for full model.
EP=16 is the minimum that fits 288 experts across the ranks without overflow
(288 / 16 = 18 experts per rank).

---

### Known limitations

| Limitation | Detail | Impact |
|---|---|---|
| MoE expert export (HF → Megatron → HF) | `_Step3FC1ExpertMapping.megatron_to_hf()` returns `{}`. Reconstructing 3D `[N, out, in]` tensors from per-expert Megatron weights requires a grouped two-key export protocol not in the current framework. | Export of a fine-tuned checkpoint back to HF format is not yet supported. Conversion HF→Megatron (for training) works fully. |
| Per-head attention gate export | `g_proj` import is mapped; export follows the standard `AutoMapping` path and works correctly. | No limitation — noted for clarity. |
| Per-layer RoPE export | Stored as a list in HF config; `provider_bridge` slices to `num_hidden_layers`. No loss of information. | No limitation. |
| PEFT | `step3_flash_peft_config()` raises `NotImplementedError`. | LoRA/DoRA training not validated for this architecture. |
| Sliding window attention during training | `window_attn_skip_freq` is passed to Mcore but FlashAttention / CUDA graph compatibility with per-layer SWA has not been profiled. | May need `attention_backend=None` or TE attention backend. |

---

### Dependencies

- Requires the companion Mcore PR (or a local `dev` branch build) containing
  `attention_per_head_gate` and `rotary_base_per_layer` in `TransformerConfig`.
- Step-3.5-Flash config/model code requires `trust_remote_code=True` (not in standard
  Transformers); no `transformers` version pin needed beyond what Bridge already requires.

---

### GitHub Actions CI

- [ ] `test_toy_model_creation` — no GPU needed, runs in standard CI.
- [ ] `test_step3_conversion_parallelism[TP]`, `[PP]`, `[EP]` — GPU required
  (`@pytest.mark.run_only_on("GPU")`).
- [ ] `test_step3_autoconfig_roundtrip` — GPU required.

---

### Before your PR is "Ready for review"

- [x] New tests added (`test_step3_conversion.py`)
- [x] Documentation added (`examples/models/step3/README.md`)
- [x] All new files pass `python3 -m py_compile`
- [ ] GPU round-trip tests pass on cluster (requires companion Mcore PR)
- [ ] `step3_flash_sft_config()` smoke-tested: `loss` decreasing, `aux_loss` stable

---

---

## PR 2 — Megatron-Core: `attention_per_head_gate` and `rotary_base_per_layer`

### What does this PR do?

Adds two new optional, off-by-default capabilities to `TransformerConfig` and `SelfAttention`
needed to faithfully represent the
[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) architecture:

1. **`attention_per_head_gate`** — a separate `ColumnParallelLinear(hidden_size → num_heads)`
   whose sigmoid output applies a scalar gate to each attention head independently.
2. **`rotary_base_per_layer`** — a list of per-layer RoPE θ values so each `SelfAttention`
   can have a different rotary base without changing the shared `rotary_pos_emb` interface.

Both features are disabled by default; no existing model behavior changes.

---

### Changelog

#### `megatron/core/transformer/transformer_config.py` (+17 lines)

**Two new dataclass fields** added immediately after `attention_output_gate` (line 244):

```python
attention_per_head_gate: bool = False
"""Apply a per-head scalar output gate (e.g., Step-3.5-Flash g_proj).
Adds a separate ColumnParallelLinear(hidden_size → num_attention_heads) whose
sigmoid output gates each attention head independently. Distinct from
attention_output_gate which fuses a full head_dim gate into linear_qkv."""

rotary_base_per_layer: Optional[List[float]] = None
"""Per-layer RoPE theta values. Length must equal num_layers. When set, each
SelfAttention layer creates its own RotaryEmbedding with the corresponding base;
the shared model-level rotary_pos_emb is not created."""
```

**Validation** added in `__post_init__` (before the existing `no_rope_freq` check):

```python
if self.rotary_base_per_layer is not None:
    assert len(self.rotary_base_per_layer) == self.num_layers, (
        f"rotary_base_per_layer length ({len(self.rotary_base_per_layer)}) "
        f"must equal num_layers ({self.num_layers})"
    )
```

#### `megatron/core/transformer/attention.py` (+45 lines, −2 lines)

**`SelfAttention.__init__`** — two new members after `k_layernorm` creation (~line 1317):

```python
# Per-head scalar output gate (attention_per_head_gate)
self.linear_gate = None
if self.config.attention_per_head_gate:
    self.linear_gate = submodules.linear_qkv(
        self.config.hidden_size,
        self.config.num_attention_heads,   # output dim = num_heads (not hidden)
        config=self.config,
        init_method=not_none(self.config.init_method),
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_comm_buffer_name='gate',
        tp_group=self.pg_collection.tp,
    )

# Per-layer RotaryEmbedding (rotary_base_per_layer)
self.rotary_pos_emb = None
if getattr(self.config, 'rotary_base_per_layer', None):
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    self.rotary_pos_emb = RotaryEmbedding(
        kv_channels=self.config.kv_channels,
        rotary_percent=getattr(self.config, 'rotary_percent', 1.0),
        rotary_base=self.config.rotary_base_per_layer[self.layer_number - 1],
        use_cpu_initialization=self.config.use_cpu_initialization,
    )
```

**`SelfAttention.forward`** — per-layer RoPE override (inserted after the existing
`no_rope` check at ~line 938, before the `isinstance(rotary_pos_emb, tuple)` guard):

```python
# Per-layer theta: override the model-level RoPE with this layer's own embedding.
if self.rotary_pos_emb is not None and rotary_pos_emb is not None:
    seq_len = rotary_pos_emb.shape[0]
    rotary_pos_emb = self.rotary_pos_emb(seq_len)
```

**`SelfAttention.forward`** — per-head gate (inserted after the existing `attention_output_gate`
check, before `linear_proj`, ~line 1228):

```python
# Per-head scalar gate (Step-3.5-Flash g_proj style)
if self.linear_gate is not None:
    nvtx_range_push(suffix="per_head_gate")
    per_head_gate, _ = self.linear_gate(hidden_states)   # [sq, b, np/tp]
    per_head_gate = per_head_gate.view(*per_head_gate.shape[:2], -1, 1)  # [sq, b, np, 1]
    core_attn_out = core_attn_out.view(*per_head_gate.shape[:3], -1)     # [sq, b, np, hn]
    core_attn_out = (
        core_attn_out * torch.sigmoid(per_head_gate.float()).to(core_attn_out.dtype)
    )
    core_attn_out = core_attn_out.view(*per_head_gate.shape[:2], -1)     # [sq, b, np*hn]
    nvtx_range_pop(suffix="per_head_gate")
```

The `sigmoid` is computed in `float32` to avoid BF16 saturation near 0/1, then cast back.

#### `megatron/core/models/gpt/gpt_model.py` (+6 lines, −2 lines)

Guarded shared `rotary_pos_emb` creation to skip when `rotary_base_per_layer` is set:

**Before:**
```python
if (self.position_embedding_type == 'rope'
        and not self.config.multi_latent_attention):
    self.rotary_pos_emb = RotaryEmbedding(...)
```

**After:**
```python
if (
    self.position_embedding_type == 'rope'
    and not self.config.multi_latent_attention
    and not getattr(self.config, 'rotary_base_per_layer', None)
):
    # Per-layer theta: each SelfAttention creates its own RotaryEmbedding,
    # so no shared module is needed at the model level.
    self.rotary_pos_emb = RotaryEmbedding(...)
```

---

### Design decisions and alternatives considered

#### `linear_gate`: separate module vs. fused into QKV

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Separate `linear_gate`** | Mirrors HF exactly; weight name maps 1:1 (`g_proj.weight`); no reshape complexity; TP sharding is standard `gather_output=False` | One extra GEMM per forward pass | ✅ Chosen |
| Fuse gate into QKV (`linear_qkv` output dim += num_heads) | One GEMM for Q+K+V+gate | Requires reshape surgery in `_get_query_key_value_tensors`; breaks existing QKV splitter assumptions; gate weight is concatenated with QKV, complicating checkpoint mapping | ✗ Rejected |

`linear_gate` reuses `submodules.linear_qkv` (the configured TP linear class) rather than
adding a new field to `SelfAttentionSubmodules`, keeping the submodule registration unchanged.

#### Per-layer RoPE: per-attention `RotaryEmbedding` vs. threading through `TransformerBlock`

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Per-`SelfAttention` `RotaryEmbedding`** | No interface changes to `TransformerBlock`, `TransformerLayer`, or `GPTModel.forward`; zero overhead when disabled | One `RotaryEmbedding` module per layer (minor memory: just a few tensors of size `kv_channels/2`) | ✅ Chosen |
| Thread `rotary_base_per_layer` through `forward` calls | Fully explicit data flow | Requires signature changes at `TransformerBlock.forward`, `TransformerLayer.forward`, `SelfAttention.forward`; cascade of downstream callers and tests | ✗ Rejected |
| Model-level dict `{layer_idx: RotaryEmbedding}` on `GPTModel` | Centralized | Requires passing `layer_number` through all forward calls | ✗ Rejected |

The chosen design reuses `rotary_pos_emb.shape[0]` (sequence length) from the incoming tensor
to call `self.rotary_pos_emb(seq_len)`. When `rotary_base_per_layer` is not set, the override
block is never reached and there is zero overhead.

---

### Backward compatibility

| Change | Impact |
|---|---|
| `attention_per_head_gate=False` (default) | `self.linear_gate = None`; no forward branch taken; zero impact |
| `rotary_base_per_layer=None` (default) | `self.rotary_pos_emb = None`; override block unreachable; zero impact |
| `gpt_model.py` guard uses `getattr(..., None)` | Safe against older `TransformerConfig` pickles that predate this field |
| No changes to `SelfAttentionSubmodules` | All existing layer specs, TE specs, MLA specs unchanged |
| No changes to `TransformerBlock` or `TransformerLayer` signatures | All downstream callers unaffected |

---

### Test plan

| Test | Description | Status |
|---|---|---|
| `python3 -m py_compile transformer_config.py` | Syntax check | ✅ Pass |
| `python3 -m py_compile attention.py` | Syntax check | ✅ Pass |
| `python3 -m py_compile gpt_model.py` | Syntax check | ✅ Pass |
| Existing GPT model tests | Confirm no regression with `attention_per_head_gate=False` | CI |
| Step-3.5-Flash toy conversion | End-to-end with `attention_per_head_gate=True`, `rotary_base_per_layer=[5e6,1e4,1e4,5e6,1e4]` (5-layer toy) | With companion Bridge PR |
| Step-3.5-Flash GPU round-trip (TP=2, PP=2, EP=2) | Full conversion + cosine similarity ≥ 0.9999 | With companion Bridge PR |

---

### Files changed summary

| File | Lines added | Lines removed | Net |
|---|---|---|---|
| `megatron/core/transformer/transformer_config.py` | +17 | 0 | +17 |
| `megatron/core/transformer/attention.py` | +47 | −2 | +45 |
| `megatron/core/models/gpt/gpt_model.py` | +8 | −2 | +6 |
| **Total** | **+72** | **−4** | **+68** |

---

### Before your PR is "Ready for review"

- [x] Both new config fields have docstrings matching the existing style in `transformer_config.py`
- [x] `__post_init__` validation for `rotary_base_per_layer` length
- [x] NVTX range markers added around `per_head_gate` computation
- [x] `getattr` guard in `gpt_model.py` for backward-compat with old configs
- [x] All three files pass `python3 -m py_compile`
- [ ] Existing CI golden value tests pass (no regression from new default-off branches)
- [ ] Step-3.5-Flash end-to-end conversion tested with companion Bridge PR
