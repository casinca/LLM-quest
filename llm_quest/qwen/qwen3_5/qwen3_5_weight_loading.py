"""
This module loads pretrained Qwen3.5 models from Hugging Face
and converts them to work with our custom Qwen3.5 implementation.

Copy/paste from `qwen3_weight_loading.py`.
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from config import QWEN3_5_08B_CONFIG
from llm_quest.qwen.qwen3_5.qwen3_5_model import Qwen3_5Model


def get_remapping_rules():
    """Get the remapping rules for converting Hugging Face weight names to our implementation names."""
    rules = [
        # outside transformer blocks
        ("model.language_model.embed_tokens.weight", "emb_dict.weight"),
        ("model.language_model.norm.weight", "final_norm.scale"),
        ("model.language_model.layers.", "trf_blocks."),
        # norms
        (".input_layernorm.weight", ".norm1.scale"),
        (".post_attention_layernorm.weight", ".norm2.scale"),
        # Inside transformer block:
        # full_attention (GatedAttention)
        # HF q_proj is fused [query, gate] → our fused q_proj
        (".self_attn.q_proj.weight", ".att.w_queries_gate.weight"),
        (".self_attn.k_proj.weight", ".att.w_keys.weight"),
        (".self_attn.v_proj.weight", ".att.w_values.weight"),
        (".self_attn.o_proj.weight", ".att.out_proj.weight"),
        (".self_attn.q_norm.weight", ".att.q_norm.scale"),
        (".self_attn.k_norm.weight", ".att.k_norm.scale"),
        # linear_attention (FusedGatedDeltaNet)
        (".linear_attn.A_log", ".att.log_A"),
        (".linear_attn.dt_bias", ".att.dt_bias"),
        (".linear_attn.in_proj_qkv.weight", ".att.w_qkv.weight"),
        (".linear_attn.in_proj_z.weight", ".att.w_gate.weight"),
        (".linear_attn.in_proj_b.weight", ".att.w_beta.weight"),
        (".linear_attn.in_proj_a.weight", ".att.w_alpha.weight"),
        (".linear_attn.conv1d.weight", ".att.conv1d.weight"),
        (".linear_attn.norm.weight", ".att.post_norm.weight"),
        (".linear_attn.out_proj.weight", ".att.out_proj.weight"),
        # dense MLP
        (".mlp.gate_proj.weight", ".ffn.lin_gate.weight"),
        (".mlp.up_proj.weight", ".ffn.lin1.weight"),
        (".mlp.down_proj.weight", ".ffn.lin2.weight"),
    ]

    return rules


def _convert_weights(hf_state_dict, our_state_dict, remapping_rules, ignored_prefixes=None):
    """
    Convert HF weights to our implementation weights.

    Args:
    hf_state_dict (dict): Hugging Face state dictionary
    our_state_dict (dict): Our implementation state dictionary
    remapping_rules (list): Remapping rules
    """
    if ignored_prefixes is None:
        ignored_prefixes = ("model.visual.", "mtp.")  # skip vision and MTP weights TODO

    converted_weights = {}
    skipped = []

    for hf_name, hf_weight in hf_state_dict.items():
        # skip vision/MTP weights
        if any(hf_name.startswith(prefix) for prefix in ignored_prefixes):
            skipped.append(hf_name)
            continue

        our_name = hf_name
        for pattern, replacement in remapping_rules:
            if pattern in our_name:
                our_name = our_name.replace(pattern, replacement)
                if pattern == hf_name:
                    break

        if our_name in our_state_dict:
            if hf_weight.shape == our_state_dict[our_name].shape:
                converted_weights[our_name] = hf_weight.clone()
            else:
                print(
                    f"WARNING: Shape mismatch: {our_name}: HF {hf_weight.shape} vs Ours {our_state_dict[our_name].shape}"
                )
        else:
            print(f"WARNING: No match for HF weight '{hf_name}' → tried '{our_name}'")

    if skipped:
        print(f"Skipped {len(skipped)} weights (vision/MTP)")

    return converted_weights


def load_qwen3_5_weights(model, model_cfg):
    """
    Download Qwen3.5 weights from HF and convert + load to our implementation.

    returns:
        model: our Qwen3.5 model with corresponding weights loaded
    """
    hf_model_name = model_cfg["model_path"]
    print(f"Loading {hf_model_name} from Hugging Face...")

    #########################
    ### Download weights ###
    try:
        print("Downloading weights...")
        repo_dir = snapshot_download(repo_id=hf_model_name)

        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)

            hf_state_dict = {}
            for filename in set(index["weight_map"].values()):
                shard_path = os.path.join(repo_dir, filename)
                shard = load_file(shard_path)
                hf_state_dict.update(shard)
            print(f"Successfully loaded weights from {hf_model_name}")
        else:
            weights_file = hf_hub_download(repo_id=hf_model_name, filename="model.safetensors")
            hf_state_dict = load_file(weights_file)
            print(f"Successfully loaded weights from {hf_model_name}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Get remapping rules and our model state dict
    remapping_rules = get_remapping_rules()
    our_state_dict = model.state_dict()

    ########################
    ### Convert weights ###
    print("\nConverting weights...")
    converted_weights = _convert_weights(hf_state_dict, our_state_dict, remapping_rules)

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

    print("\nTie_embeddings=True, trying weight tying...")
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
    """Report loading results."""
    print(f"Loaded {len(converted_weights)}/{len(model.state_dict())} weights successfully\n")

    if load_result.missing_keys:
        print(f"Missing keys ({len(load_result.missing_keys)}):")
        print("-> lm_head/out_head expected to be missing with tie_embeddings=True")
        print("-> custom buffers expected to be missing, ex: mask, cos, sin\n")

    if load_result.unexpected_keys:
        print(f"Unexpected keys: {load_result.unexpected_keys}")


def test_generation_with_weights(device="cuda"):
    """
    Test the loaded model with a simple generation
    """
    print("\n=== Testing Generation ===")

    model_cfg = QWEN3_5_08B_CONFIG
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])

    model = Qwen3_5Model(model_cfg)
    model = load_qwen3_5_weights(model=model, model_cfg=model_cfg)
    model.to(device).eval()

    prompt = "Give me a short introduction to large language models."
    print(f"Prompt: '{prompt}'\n")

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
