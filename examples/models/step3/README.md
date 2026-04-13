# Step-3.5-Flash Examples

This directory contains example scripts for [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash), a 196.81B sparse MoE model (~11B active parameters) with 45 transformer layers (3 dense + 42 MoE), 288 routed experts + 1 shared expert per MoE layer, and top-8 sigmoid routing.

> **Note**: Step-3.5-Flash uses a custom `step3p5` architecture not part of the standard Transformers library.
> All commands require `--trust-remote-code`.

## Hardware Requirements

Step-3.5-Flash conversion and inference run on **1 node × 8 GPUs** (H100/A100 80 GB):

| Component | Per-GPU at EP=8 |
|---|---|
| Routed expert weights (36 experts/rank) | ~47 GB |
| Non-expert weights (replicated) | ~10 GB |
| **Total model weights** | **~57 GB** |
| Remaining headroom (80 GB GPU) | ~23 GB (KV cache / buffers) |

- EP=8 distributes 288 experts across 8 ranks → 36 experts per rank.
- TP does **not** reduce expert memory — use EP instead.
- Recommended config for 8 GPUs: `TP=1, PP=1, EP=8`.

## Checkpoint Conversion

[slurm_conversion.sh](slurm_conversion.sh) sweeps multiple TP/PP/EP configs to verify HF ↔ Megatron round-trip conversion.

### Setup

Edit the variables at the top of `slurm_conversion.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
# Optional:
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/path/to/shared/HF_HOME"
```

### Submit

```bash
sbatch examples/models/step3/slurm_conversion.sh
```

### Expected output (per config)

```
======================================
Step-3.5-Flash Round-Trip Conversion Sweep (1 node × 8 GPUs)
Job: <JOB_ID> | Nodes: 1
Parallelism configs: 1,1,8 2,1,4 1,2,4
======================================
Config 1/3: TP=1, PP=1, EP=8
...
All parameters match: True ✅
[OK] Config 1: TP=1, PP=1, EP=8 passed
...
All 3 configs passed
======================================
```

## Training

[train_step3_flash.py](train_step3_flash.py) supports both pre-training and SFT using the recipe
helpers in `megatron.bridge.recipes.step3`.  [slurm_train.sh](slurm_train.sh) wraps it for
multi-node Slurm submission.

### Step 1 — Convert the HF checkpoint (once)

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model stepfun-ai/Step-3.5-Flash \
    --trust-remote-code \
    --megatron-path /path/to/megatron_ckpt \
    --tp 1 --pp 1 --ep 8
```

Or use [slurm_conversion.sh](slurm_conversion.sh) for a multi-node round-trip verification.

### Step 2 — Edit slurm_train.sh

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
PRETRAINED_CHECKPOINT="/path/to/megatron_ckpt"   # from Step 1
DATA_PATH="/path/to/dataset.jsonl"               # JSONL for SFT
SAVE_DIR="/path/to/training_checkpoints"
```

Optional: edit [conf/step3_flash_sft_override.yaml](conf/step3_flash_sft_override.yaml) to tune
training hyperparameters (batch size, learning rate, iterations, etc.).

### Step 3 — Submit

```bash
# SFT (default):
sbatch examples/models/step3/slurm_train.sh

# Pre-training from scratch or from an existing checkpoint:
TRAIN_MODE=pretrain sbatch examples/models/step3/slurm_train.sh
```

### CLI overrides

Any `key=value` argument after `--` is passed as a Hydra-style dot-notation override:

```bash
torchrun --nproc_per_node=8 examples/models/step3/train_step3_flash.py \
    --mode sft \
    --pretrained-checkpoint /path/to/ckpt \
    model.expert_model_parallel_size=32 \
    train.train_iters=500
```

### Verifying correctness

Check these indicators during the first few hundred steps:

| Signal | Healthy range |
|---|---|
| `lm loss` | Decreasing; starting around 5–9 for SFT on real data |
| `load balancing loss` | Small positive value (~1e-3 scale); not diverging |
| `grad norm` | < 1.0 after warmup; spikes > 10 suggest LR is too high |
| `tokens per GPU per sec` | Stable; big drops indicate PP bubble or expert imbalance |
| Expert routing | Enable `--log-moe-router-stats` (or check `aux_loss` curve) |

If `lm loss` stays flat or NaN appears within the first 10 steps, check:
- Checkpoint was loaded correctly (`pretrained_checkpoint` path is set)
- `trust_remote_code=True` is active (already set in the recipe)
- All ranks have the same `seed` (default: `5678` in SFT recipe)

## Inference

[slurm_inference.sh](slurm_inference.sh) runs text generation on the full checkpoint with `TP=1, PP=1, EP=8` (1 node × 8 GPUs).

### Setup

Edit the variables at the top of `slurm_inference.sh`:

```bash
CONTAINER_IMAGE="/path/to/container.sqsh"
export HF_TOKEN="hf_your_token_here"
```

### Submit

```bash
sbatch examples/models/step3/slurm_inference.sh
```

### Expected output

```
======================================
Step-3.5-Flash Inference (1 node × 8 GPUs)
Job: <JOB_ID> | Nodes: 1
TP=1 PP=1 EP=8 (Total GPUs: 8)
======================================
...
======== GENERATED TEXT OUTPUT ========
Prompt: What is artificial intelligence?
Generated: What is artificial intelligence? Artificial intelligence (AI) refers to...
=======================================
```
