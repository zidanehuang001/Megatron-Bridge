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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import bf16_mixed

_HF_MODEL_ID = "stepfun-ai/Step-3.5-Flash"


def step3_flash_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Step-3.5-Flash (196.81B total / ~11B active).

    Step-3.5-Flash is a sparse MoE model with 45 layers (3 dense + 42 MoE),
    288 routed experts + 1 shared expert per MoE layer, top-8 sigmoid routing,
    and zero-centered RMSNorm.

    Recommended parallelism: TP=1, PP=4, EP=16.
      - 288 routed experts / EP=16 = 18 experts per GPU rank.
      - Adjust PP and EP based on available GPU memory and count.

    Note:
        The model requires ``trust_remote_code=True`` because it uses a custom
        ``step3p5`` architecture that is not part of the standard Transformers library.

    Returns:
        ConfigContainer with all settings pre-configured for Step-3.5-Flash pre-training.
    """
    cfg = _pretrain_common()

    # Model config — loads architecture from HF without downloading weights
    cfg.model = AutoBridge.from_hf_pretrained(
        _HF_MODEL_ID, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _HF_MODEL_ID

    # Dataset config — mock data by default
    cfg.dataset.blend = None  # Set to dataset path(s) for real training
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    # ── Parallelism ───────────────────────────────────────────────────────────
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16  # 288 experts / 16 EP = 18 per rank
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # ── MoE token dispatcher ──────────────────────────────────────────────────
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # ── Training ──────────────────────────────────────────────────────────────
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # ── Transformer Engine ────────────────────────────────────────────────────
    cfg.model.transformer_impl = "transformer_engine"

    # ── CUDA Graph ────────────────────────────────────────────────────────────
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # ── Kernel selections ─────────────────────────────────────────────────────
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # ── Memory saving (recompute) ─────────────────────────────────────────────
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # ── FP8 (disabled by default; enable for FP8 training) ───────────────────
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # ── Optimizer ─────────────────────────────────────────────────────────────
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # ── MoE overlap ───────────────────────────────────────────────────────────
    cfg.model.moe_shared_expert_overlap = False

    # ── DDP ───────────────────────────────────────────────────────────────────
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    # ── MoE Force Load Balancing ──────────────────────────────────────────────
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


def step3_flash_sft_config() -> ConfigContainer:
    """Return an SFT config for Step-3.5-Flash (196.81B total / ~11B active).

    Recommended parallelism: TP=1, PP=2, EP=16.

    Note:
        The model requires ``trust_remote_code=True`` because it uses a custom
        ``step3p5`` architecture.

    Returns:
        ConfigContainer with all settings pre-configured for Step-3.5-Flash SFT.
    """
    cfg = _sft_common()

    # Dataset — packed sequence for efficient SFT
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=True, pad_seq_to_mult=1)

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained(
        _HF_MODEL_ID, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = _HF_MODEL_ID

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # ── Parallelism ───────────────────────────────────────────────────────────
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # ── MoE token dispatcher ──────────────────────────────────────────────────
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # ── Mixed precision ───────────────────────────────────────────────────────
    cfg.mixed_precision = bf16_mixed()

    # ── Training ──────────────────────────────────────────────────────────────
    cfg.train.train_iters = 1000
    cfg.validation.eval_interval = 30
    cfg.validation.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # ── Logger ────────────────────────────────────────────────────────────────
    cfg.logger.log_interval = 1

    # ── Optimizer ─────────────────────────────────────────────────────────────
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # ── Scheduler ─────────────────────────────────────────────────────────────
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.max_lr = 5e-6
    cfg.optimizer.min_lr = 5e-6

    # ── Transformer Engine ────────────────────────────────────────────────────
    cfg.model.transformer_impl = "transformer_engine"

    # ── CUDA Graph ────────────────────────────────────────────────────────────
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # ── Kernel selections ─────────────────────────────────────────────────────
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # ── Memory saving ─────────────────────────────────────────────────────────
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # ── FP8 (disabled by default) ─────────────────────────────────────────────
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    cfg.model.moe_router_padding_for_fp8 = False

    # ── MoE overlap ───────────────────────────────────────────────────────────
    cfg.model.moe_shared_expert_overlap = False

    # ── Checkpoint ────────────────────────────────────────────────────────────
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/megatron/checkpoint"

    # ── DDP ───────────────────────────────────────────────────────────────────
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_megatron_fsdp = False

    # ── MoE Force Load Balancing ──────────────────────────────────────────────
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    cfg.rng.seed = 5678

    return cfg


def step3_flash_peft_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return a PEFT (LoRA/DoRA) config for Step-3.5-Flash.

    Note:
        PEFT is not currently validated for Step-3.5-Flash. If you encounter issues,
        fall back to full SFT via ``step3_flash_sft_config()``.

    Args:
        peft_scheme: PEFT scheme — ``"lora"``, ``"dora"``, or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Step-3.5-Flash PEFT.
    """
    raise NotImplementedError(
        "PEFT is not currently supported for Step-3.5-Flash. "
        "Use full SFT via step3_flash_sft_config() instead."
    )
