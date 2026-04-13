#!/usr/bin/env python3
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

"""
Step-3.5-Flash Training Script with YAML and CLI Configuration Overrides.

Supports both pre-training (from random init or a Megatron checkpoint) and
supervised fine-tuning (SFT).  Uses the recipe helpers in
``megatron.bridge.recipes.step3``.

Examples:
    1. Convert an HF checkpoint to Megatron format (run once before training):
         $ python examples/conversion/convert_checkpoints.py import \\
               --hf-model stepfun-ai/Step-3.5-Flash \\
               --trust-remote-code \\
               --megatron-path /path/to/megatron_ckpt \\
               --tp 1 --pp 4 --ep 16

    2. Pre-train (from scratch / from Megatron checkpoint):
         $ torchrun --nproc_per_node=8 examples/models/step3/train_step3_flash.py \\
               --mode pretrain \\
               --pretrained-checkpoint /path/to/megatron_ckpt

    3. SFT:
         $ torchrun --nproc_per_node=8 examples/models/step3/train_step3_flash.py \\
               --mode sft \\
               --pretrained-checkpoint /path/to/megatron_ckpt \\
               --data-path /path/to/dataset.jsonl

    4. CLI overrides (Hydra-style dot notation):
         $ torchrun --nproc_per_node=8 examples/models/step3/train_step3_flash.py \\
               --mode sft \\
               model.tensor_model_parallel_size=1 \\
               model.pipeline_model_parallel_size=4 \\
               model.expert_model_parallel_size=16 \\
               train.train_iters=1000

    5. YAML override file:
         $ torchrun --nproc_per_node=8 examples/models/step3/train_step3_flash.py \\
               --config-file examples/models/step3/conf/step3_flash_sft_override.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.step3 import step3_flash_pretrain_config, step3_flash_sft_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.training.vlm_step import forward_step
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).parent.resolve()


def parse_cli_args() -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description="Train Step-3.5-Flash with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft"],
        default="sft",
        help="Training mode: 'pretrain' (GPT-style) or 'sft' (supervised fine-tuning). Default: sft",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a YAML OmegaConf override file.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to an imported Megatron checkpoint directory to load before training. "
            "Generate it with: python examples/conversion/convert_checkpoints.py import"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to a JSONL dataset file (SFT mode) or a data prefix / blend spec "
            "(pretrain mode, e.g. '1.0 /path/to/data_prefix')."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    args, cli_overrides = parse_cli_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info("Megatron-Bridge Step-3.5-Flash Training Script")
    logger.info(f"Mode: {args.mode}")

    # ── Load base recipe config ───────────────────────────────────────────────
    if args.mode == "pretrain":
        cfg: ConfigContainer = step3_flash_pretrain_config()
    else:
        cfg = step3_flash_sft_config()

    # ── Apply checkpoint / dataset paths from CLI ─────────────────────────────
    if args.pretrained_checkpoint is not None:
        cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
        logger.info(f"Loading pretrained checkpoint from: {args.pretrained_checkpoint}")

    if args.data_path is not None:
        if args.mode == "sft":
            cfg.dataset.dataset_name = "json"
            cfg.dataset.dataset_kwargs = cfg.dataset.dataset_kwargs or {}
            cfg.dataset.dataset_kwargs["data_files"] = args.data_path
        else:
            cfg.dataset.blend = args.data_path
        logger.info(f"Dataset path: {args.data_path}")

    if get_rank_safe() == 0:
        cfg.print_yaml()

    # ── Merge YAML overrides ───────────────────────────────────────────────────
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        logger.info(f"Applying YAML overrides from: {args.config_file}")
        yaml_overrides = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides)

    # ── Merge CLI dot-notation overrides ──────────────────────────────────────
    if cli_overrides:
        logger.info(f"Applying CLI overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_cfg_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_cfg_dict, excluded_fields)

    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
