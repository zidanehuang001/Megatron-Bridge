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
# Step-3.5-Flash Inference (Single-Node via Slurm)
#
# Step-3.5-Flash (MoE: 288 experts, top-8, 196.81B total / ~11B active)
# Runs on 1 node × 8 GPUs (H100/A100 80 GB recommended).
#
# EP=8 distributes 288 experts across 8 ranks (36 experts/rank, ~58 GB/GPU).
# TP does NOT reduce expert memory — use EP instead.
#
# Usage:
#   1. Fill in CONTAINER_IMAGE, CONTAINER_MOUNTS, and token exports
#   2. Submit: sbatch examples/models/step3/slurm_inference.sh
# ==============================================================================

#SBATCH --job-name=step3-flash-inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --account=<your-account>
#SBATCH --partition=batch
#SBATCH --output=logs/step3_flash_inference_%j.log
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

# ── Model / Parallelism ──────────────────────────────────────────────────
MODEL_NAME=Step-3.5-Flash
HF_MODEL_ID=stepfun-ai/$MODEL_NAME
PROMPT="What is artificial intelligence?"
MAX_NEW_TOKENS=100
TP=1
PP=1
EP=8

# ── Environment ───────────────────────────────────────────────────────────
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Step-3.5-Flash Inference (1 node × 8 GPUs)"
echo "Job: $SLURM_JOB_ID | Nodes: $SLURM_JOB_NUM_NODES"
echo "TP=$TP PP=$PP EP=$EP (Total GPUs: $((TP * PP * EP)))"
echo "======================================"

mkdir -p logs

if [ -z "$CONTAINER_IMAGE" ]; then
    echo "ERROR: CONTAINER_IMAGE must be set."
    exit 1
fi

SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"
if [ -n "$CONTAINER_MOUNTS" ]; then
    SRUN_CMD="$SRUN_CMD --container-mounts=$CONTAINER_MOUNTS"
fi

# Sync dependencies once per node, then run inference
CMD="if [ \"\$SLURM_LOCALID\" -eq 0 ]; then uv sync; else sleep 10; fi && "
CMD="${CMD}uv run --no-sync python examples/conversion/hf_to_megatron_generate_text.py"
CMD="$CMD --hf_model_path $HF_MODEL_ID"
CMD="$CMD --prompt '$PROMPT'"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --tp $TP --pp $PP --ep $EP"
CMD="$CMD --trust_remote_code"

echo "Executing: $CMD"

$SRUN_CMD bash -c "cd $WORKDIR && $CMD"

echo "======================================"
echo "Inference completed"
echo "======================================"
