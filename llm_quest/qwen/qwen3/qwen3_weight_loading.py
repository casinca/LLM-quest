"""
This module loads pretrained Qwen3 models from Hugging Face
and convert them to work with our custom Qwen3 implementation.
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import qwen3_config_creator
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model


def get_remapping_rules(model_cfg):
    """
    Get the remapping rules for converting Hugging Face weight names to our implementation names.
    Supports both dense and MoE models.
    """
    rules = [
        # Exact matches for outside transformer block weights:
        ("model.embed_tokens.weight", "emb_dict.weight"),
        ("model.norm.weight", "final_norm.weight"),
        ("model.layers.", "trf_blocks."),
        # Inside transformer block:
        # attention blocks
        (".self_attn.q_proj.weight", ".att.w_queries.weight"),
        (".self_attn.k_proj.weight", ".att.w_keys.weight"),
        (".self_attn.v_proj.weight", ".att.w_values.weight"),
        (".self_attn.o_proj.weight", ".att.out_proj.weight"),
        (".self_attn.q_norm.weight", ".att.q_norm.weight"),
        (".self_attn.k_norm.weight", ".att.k_norm.weight"),
        # norms
        (".input_layernorm.weight", ".norm1.weight"),
        (".post_attention_layernorm.weight", ".norm2.weight"),
    ]

    # weight tying
    if not model_cfg["tie_embeddings"]:
        rules.append(("lm_head.weight", "out_head.weight"))

    # MoE vs Dense
    if model_cfg["model_type"] == "moe":
        moe_rules = [
            # MoE router/gate
            (".mlp.gate.weight", ".moe.gate.weight"),
            # expert MLP layers
            (".mlp.experts.", ".moe.experts."),
            (".gate_proj.weight", ".lin_gate.weight"),
            (".up_proj.weight", ".lin1.weight"),
            (".down_proj.weight", ".lin2.weight"),
        ]
        rules.extend(moe_rules)
    else:
        dense_rules = [
            # dense MLP layers
            (".mlp.gate_proj.weight", ".ffn.lin_gate.weight"),
            (".mlp.up_proj.weight", ".ffn.lin1.weight"),
            (".mlp.down_proj.weight", ".ffn.lin2.weight"),
        ]
        rules.extend(dense_rules)

    return rules


def _convert_weights(hf_state_dict, our_state_dict, remapping_rules):

        ### Dense vs MoE ###
        if model_type == "moe":
            moe_weights = {
                # Router/gate weight
                f"{layer_prefix_hf}.mlp.gate.weight": f"{layer_prefix_ours}.moe.gate.weight",
            }

            # Expert weights
            for expert_idx in range(model_cfg["num_experts"]):
                expert_weights = {
                    f"{layer_prefix_hf}.mlp.experts.{expert_idx}.gate_proj.weight": f"{layer_prefix_ours}.moe.experts.{expert_idx}.lin_gate.weight",
                    f"{layer_prefix_hf}.mlp.experts.{expert_idx}.up_proj.weight": f"{layer_prefix_ours}.moe.experts.{expert_idx}.lin1.weight",
                    f"{layer_prefix_hf}.mlp.experts.{expert_idx}.down_proj.weight": f"{layer_prefix_ours}.moe.experts.{expert_idx}.lin2.weight",
                }
                moe_weights.update(expert_weights)
            weight_mapping.update(moe_weights)

        else:  # Dense FFN
            ffn_weights = {
                f"{layer_prefix_hf}.mlp.gate_proj.weight": f"{layer_prefix_ours}.ffn.lin_gate.weight",
                f"{layer_prefix_hf}.mlp.up_proj.weight": f"{layer_prefix_ours}.ffn.lin1.weight",
                f"{layer_prefix_hf}.mlp.down_proj.weight": f"{layer_prefix_ours}.ffn.lin2.weight",
            }
            weight_mapping.update(ffn_weights)

    return weight_mapping


def load_qwen3_weights(model, model_cfg):
    """
    Download Qwen3 weights from Hugging Face and convert + load to our implementation.
    Supports both dense models (single file) and larger/MoE models (multi-file).

    returns:
        model: our Qwen3 model with the corresponding weights loaded
    """

    hf_model_name = model_cfg["model_path"]
    model_type = model_cfg["model_type"]
    print(f"Loading {hf_model_name} ({model_type}) from Hugging Face...")

    #########################
    ### Download weights ###
    try:
        print("Downloading weights file...")
        # snapshot_download for both large or MoE to handle multi-file models
        # this part of the code is from @rasbt for loading sharded weights
        repo_dir = snapshot_download(repo_id=hf_model_name)

        # Load model index to find all weight files
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

            # Load weights from all shards
            hf_state_dict = {}
            for filename in set(index["weight_map"].values()):
                shard_path = os.path.join(repo_dir, filename)
                shard = load_file(shard_path)
                hf_state_dict.update(shard)
            print(f"Successfully loaded weights from {hf_model_name}")

        else:  # fallback for small models without sharding
            weights_file = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors")
            hf_state_dict = load_file(weights_file)
            print(f"Successfully loaded weights from {hf_model_name}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Get remapping rules and our model state dict
    remapping_rules = get_remapping_rules(model_cfg=model_cfg)
    our_state_dict = model.state_dict()

    ########################
    ### Convert weights ###
    print("\nConverting weights...")
    converted_weights = {}

    for hf_name, hf_weight in hf_state_dict.items():
        if hf_name in weight_mapping:
            our_name = weight_mapping[hf_name]

            if our_name in our_state_dict:
                if hf_weight.shape == our_state_dict[our_name].shape:
                    converted_weights[our_name] = hf_weight.clone()
                else:
                    print(
                        f"WARNING: Shape mismatch:{our_name}: HF {hf_weight.shape} vs Ours {our_state_dict[our_name].shape}"
                    )

    #####################
    ### Load weights ###
    with torch.no_grad():
        load_result = model.load_state_dict(converted_weights, strict=False)
        _handle_weight_tying(model)

    _report_loading_status(model, load_result, converted_weights)

    return model


def _handle_weight_tying(model):
    """Handle weight tying after loading pretrained weights."""
    if not (hasattr(model, "tie_embeddings") and model.tie_embeddings):
        print("Tie_embeddings=False, skipping weight tying\n")
        return

    print("\nTie_embeddings=True, trying for weight tying...")
    # verify shapes match
    emb_shape, out_shape = model.emb_dict.weight.shape, model.out_head.weight.shape
    if emb_shape != out_shape:
        print(f"WARNING: Shape mismatch for weight tying: {emb_shape} vs {out_shape}")
        return

    model.out_head.weight = model.emb_dict.weight

    # sanity check
    if id(model.emb_dict.weight) == id(model.out_head.weight):
        print("Weight tied successfully\n")
    else:
        print("WARNING: Weight tying failed!\n")


def _report_loading_status(model, load_result, converted_weights):
    """Report missing keys"""
    print(f"Loaded {len(converted_weights)}/{len(model.state_dict())} weights successfully\n")

    if load_result.missing_keys:
        print(f"Missing keys: {load_result.missing_keys}")
        print("-> lm_head/out_head expected to be missing with: tie_embeddings=True")
        print("-> custom buffers expected to be missing, ex: mask, cos, sin\n")

    if load_result.unexpected_keys:
        print(f"Unexpected keys: {load_result.unexpected_keys}")


def test_generation_with_weights(device="cuda"):
    """
    Test the loaded model with a simple generation
    """
    print("\n=== Testing Generation ===")

    model_cfg = qwen3_config_creator("0.6B-Base")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])

    model = Qwen3Model(model_cfg)
    model = load_qwen3_weights(model=model, model_cfg=model_cfg)
    model.to(device).eval()

    prompt = "Give me a short introduction to large language models."  # same prompt as HF + @rasbt impl for comparison
    print(f"Prompt: '{prompt}'\n")

    # simple greedy generation
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    for _ in range(20):
        with torch.no_grad():
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    torch.manual_seed(123)
    test_generation_with_weights()
