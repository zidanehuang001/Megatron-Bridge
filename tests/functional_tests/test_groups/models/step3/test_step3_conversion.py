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

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


try:
    # Step-3.5-Flash uses a custom architecture (trust_remote_code=True).
    # Fetch the config class from HF Hub so we can instantiate a toy model
    # without downloading the full checkpoint (~400 GB).
    _step3_config_cls = AutoConfig.from_pretrained(
        "stepfun-ai/Step-3.5-Flash", trust_remote_code=True
    ).__class__
except Exception:
    pytest.skip(
        "Step-3.5-Flash config not available (requires HF hub access and trust_remote_code).",
        allow_module_level=True,
    )


# Toy config: reduced dims for fast testing.
# Keeps key architectural properties: mixed dense/MoE layers, 3D-batched MoELinear,
# shared expert, sigmoid routing with expert bias, zero-centered RMSNorm, QK norm.
HF_STEP3_TOY_MODEL_CONFIG = {
    "architectures": ["Step3p5ForCausalLM"],
    "model_type": "step3p5",
    "hidden_size": 512,
    "intermediate_size": 1024,     # Dense MLP FFN dim (layers 0-2)
    "moe_intermediate_size": 256,  # MoE routed expert FFN dim
    "num_hidden_layers": 5,        # 3 dense (0-2) + 2 MoE (3-4)
    "num_attention_heads": 8,
    "num_attention_groups": 2,     # KV heads (GQA)
    "head_dim": 64,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    # Per-layer RoPE theta (5 values for 5 layers — exercises rotary_base_per_layer)
    "rope_theta": [5000000.0, 10000.0, 10000.0, 5000000.0, 10000.0],
    # Alternating SWA pattern: layers 0-2 dense/full, layer 3 SWA, layer 4 full
    "layer_types": [
        "full_attention", "full_attention", "full_attention",
        "sliding_attention", "full_attention",
    ],
    "sliding_window": 512,
    # Per-head scalar attention gate
    "use_head_wise_attn_gate": True,
    "moe_layers_enum": "3,4",      # Which layers are MoE (0-indexed)
    "moe_num_experts": 4,          # Routed experts (288 in real model)
    "moe_top_k": 2,                # Top-K routing (8 in real model)
    "share_expert_dim": 256,       # Shared expert FFN dim
    "use_moe_router_bias": True,   # Expert bias correction for load balancing
    "vocab_size": 1024,
    "tie_word_embeddings": False,
    "attention_dropout": 0.0,
    "torch_dtype": "bfloat16",
}


class TestStep3Conversion:
    """
    Test Step-3.5-Flash conversion with different parallelism configurations.
    Uses a toy model (5 layers: 3 dense + 2 MoE, 4 experts) with random weights.
    """

    @pytest.fixture(scope="class")
    def toy_model_path(self, tmp_path_factory):
        """Create and save a toy Step-3.5-Flash model to a temporary directory."""
        temp_dir = tmp_path_factory.mktemp("step3_toy_model")
        model_dir = temp_dir / "step3_toy"

        # Load the real config class (fetched at module level), then override with toy dims.
        config = AutoConfig.from_pretrained(
            "stepfun-ai/Step-3.5-Flash", trust_remote_code=True
        )
        for key, value in HF_STEP3_TOY_MODEL_CONFIG.items():
            setattr(config, key, value)
        config.torch_dtype = torch.bfloat16

        # Instantiate with random weights.
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).bfloat16()

        model.save_pretrained(model_dir, safe_serialization=True)

        # Save a surrogate tokenizer (gpt2) since the toy model has no real tokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.save_pretrained(model_dir)
        except Exception:
            pass

        # Overwrite config.json with the exact toy config to avoid stale HF fields.
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(HF_STEP3_TOY_MODEL_CONFIG, f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, toy_model_path):
        """Verify the toy model is created correctly and has the expected structure."""
        model_path = Path(toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            sharded = list(model_path.glob("model-*-of-*.safetensors"))
            assert len(sharded) > 0, "No model weight files found"

        with open(config_file) as f:
            cfg = json.load(f)

        assert cfg["model_type"] == "step3p5"
        assert cfg["hidden_size"] == 512
        assert cfg["num_hidden_layers"] == 5
        assert cfg["moe_num_experts"] == 4
        assert isinstance(cfg["rope_theta"], list), "rope_theta should be a per-layer list"
        assert len(cfg["rope_theta"]) == cfg["num_hidden_layers"]
        assert cfg["use_head_wise_attn_gate"] is True
        assert "layer_types" in cfg and len(cfg["layer_types"]) == cfg["num_hidden_layers"]

        model = AutoModelForCausalLM.from_pretrained(
            toy_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        assert len(model.model.layers) == 5

        # Layers 0-2 are dense: expect standard mlp attribute.
        dense_layer = model.model.layers[0]
        assert hasattr(dense_layer, "mlp"), (
            f"Dense layer missing 'mlp'; got: {list(dense_layer._modules.keys())}"
        )

        # Layers 3-4 are MoE: expect moe attribute with gate_proj/up_proj/down_proj tensors.
        moe_layer = model.model.layers[3]
        assert hasattr(moe_layer, "moe"), (
            f"MoE layer missing 'moe'; got: {list(moe_layer._modules.keys())}"
        )
        moe_block = moe_layer.moe
        assert hasattr(moe_block, "gate_proj"), (
            f"MoE block missing 'gate_proj'; got: {list(moe_block._modules.keys())}"
        )
        # gate_proj is a 3D batched tensor [num_experts, intermediate_size, hidden_size]
        assert moe_block.gate_proj.weight.ndim == 3, (
            f"Expected 3D expert weight, got shape {moe_block.gate_proj.weight.shape}"
        )
        assert moe_block.gate_proj.weight.shape[0] == HF_STEP3_TOY_MODEL_CONFIG["moe_num_experts"]

        # All layers should have g_proj (per-head attention gate).
        attn = model.model.layers[0].self_attn
        assert hasattr(attn, "g_proj"), (
            f"Attention missing 'g_proj'; got: {list(attn._modules.keys())}"
        )
        assert attn.g_proj.weight.shape == (
            HF_STEP3_TOY_MODEL_CONFIG["num_attention_heads"],
            HF_STEP3_TOY_MODEL_CONFIG["hidden_size"],
        ), f"Unexpected g_proj shape: {attn.g_proj.weight.shape}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,test_name",
        [
            (2, 1, 1, "TP"),
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    def test_step3_conversion_parallelism(self, toy_model_path, tmp_path, tp, pp, ep, test_name):
        """Test Step-3.5-Flash HF→Megatron→HF round-trip with different parallelism configs."""
        test_output_dir = tmp_path / f"step3_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        repo_root = "/opt/Megatron-Bridge"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            f"--data-file={repo_root}/.coverage",
            f"--source={repo_root}/",
            "--parallel-mode",
            f"{repo_root}/examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--trust-remote-code",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(
                f"Step-3.5-Flash {test_name} conversion failed with return code {result.returncode}"
            )

        model_name = Path(toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists(), f"Converted model directory not found at {converted_dir}"

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "step3p5"
        assert saved_config["hidden_size"] == 512
        assert saved_config["moe_num_experts"] == 4

    @pytest.mark.run_only_on("GPU")
    def test_step3_autoconfig_roundtrip(self, toy_model_path, tmp_path):
        from tests.functional_tests.utils import autoconfig_roundtrip

        autoconfig_roundtrip(toy_model_path, tmp_path)
