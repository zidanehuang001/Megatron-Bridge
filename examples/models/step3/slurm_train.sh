#!/bin/bash
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

# ==============================================================================
# Step-3.5-Flash Training (Multi-Node via Slurm)
#
# Step-3.5-Flash (MoE: 288 experts, top-8, 196.81B total / ~11B active)
# Requires at least 8 nodes (64 GPUs):
#   TP=1, PP=4, EP=16 → 288/16=18 experts per rank, 4 pipeline stages.
#
# Workflow:
#   1. Convert HF checkpoint to Megatron format (run slurm_conversion.sh first).
#   2. Fill in the variables in this script.
#   3. Submit: sbatch examples/models/step3/slurm_train.sh
#
# Usage:
#   SFT (default):
#     sbatch examples/models/step3/slurm_train.sh
#   Pre-train:
#     TRAIN_MODE=pretrain sbatch examples/models/step3/slurm_train.sh
# ==============================================================================

#SBATCH --job-name=step3-flash-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/step3_flash_train_%j.log
#SBATCH --exclusive

# ── Container ────────────────────────────────────────────────────────────
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS=""
# IMPORTANT: Mount BOTH repos so the 3rdparty/Megatron-LM symlink resolves
# inside the container (symlink ../../Megatron-LM -> /opt/Megatron-LM).
# CONTAINER_MOUNTS="/path/to/Megatron-Bridge:/opt/Megatron-Bridge,/path/to/Megatron-LM:/opt/Megatron-LM"
WORKDIR="/opt/Megatron-Bridge"

# ── Tokens / Caches ──────────────────────────────────────────────────────
# export HF_TOKEN="hf_your_token_here"
# export HF_HOME="/path/to/shared/HF_HOME"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"

# ── Training mode: "pretrain" or "sft" ───────────────────────────────────
TRAIN_MODE="${TRAIN_MODE:-sft}"

# ── Paths ─────────────────────────────────────────────────────────────────
# Megatron-format checkpoint produced by slurm_conversion.sh
PRETRAINED_CHECKPOINT="/path/to/megatron_checkpoint"

# Dataset:
#   SFT mode:     path to a JSONL file with conversation turns
#   Pretrain mode: data blend prefix understood by Megatron data loader
DATA_PATH="/path/to/dataset.jsonl"

# Where to save training checkpoints
SAVE_DIR="/path/to/training_checkpoints/step3_flash"

# ── Parallelism ────────────────────────────────────────────────────────────
# 8 nodes × 8 GPUs = 64 total GPUs
# TP=1, PP=4, EP=16 → 18 experts per rank (288 / 16)
TP=1
PP=4
EP=16

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
# Reduce fragmentation for large MoE models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Step-3.5-Flash Training"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "Mode: $TRAIN_MODE | TP=$TP PP=$PP EP=$EP"
echo "======================================"

mkdir -p logs "$SAVE_DIR"

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

# Sync dependencies once per node (rank 0 on each node runs uv sync;
# other local ranks wait to avoid concurrent writes to the same cache).
SYNC="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 10; fi"

TRAIN_CMD="uv run --no-sync python examples/models/step3/train_step3_flash.py"
TRAIN_CMD="$TRAIN_CMD --mode $TRAIN_MODE"
TRAIN_CMD="$TRAIN_CMD --pretrained-checkpoint $PRETRAINED_CHECKPOINT"
TRAIN_CMD="$TRAIN_CMD --data-path $DATA_PATH"
TRAIN_CMD="$TRAIN_CMD --config-file $WORKDIR/examples/models/step3/conf/step3_flash_sft_override.yaml"
# Parallelism overrides (take precedence over the YAML)
TRAIN_CMD="$TRAIN_CMD model.tensor_model_parallel_size=$TP"
TRAIN_CMD="$TRAIN_CMD model.pipeline_model_parallel_size=$PP"
TRAIN_CMD="$TRAIN_CMD model.expert_model_parallel_size=$EP"
# Checkpoint save directory
TRAIN_CMD="$TRAIN_CMD checkpoint.save=$SAVE_DIR"

echo "Executing: $TRAIN_CMD"

$SRUN_CMD bash -c "cd $WORKDIR && $SYNC && $TRAIN_CMD"
EXIT_CODE=$?

echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training FAILED (exit code $EXIT_CODE)"
fi
echo "======================================"
exit $EXIT_CODE
