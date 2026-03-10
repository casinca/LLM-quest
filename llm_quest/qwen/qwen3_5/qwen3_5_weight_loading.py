"""
This module loads pretrained Qwen3.5 models from Hugging Face
and converts them to work with our custom Qwen3.5 implementation.

Similar to `qwen3_weight_loading.py`.
"""

import torch

from config import QWEN3_5_08B_CONFIG
from llm_quest.qwen.qwen3_5.qwen3_5_text_model import Qwen3_5TextModel
from llm_quest.qwen.qwen3_5.qwen3_5_vlm_model import Qwen3_5VLM
from llm_quest.utils import (
    convert_weights,
    download_hf_weights,
    handle_weight_tying,
    report_loading_status,
    test_generation_with_weights,
)


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


def get_vision_remapping_rules():
    """Get the remapping rules for converting HF vision weight names to our implementation names."""
    rules = [
        # patch embedding
        ("model.visual.patch_embed.proj.", "patch_embed.conv_proj."),
        # positional embedding
        ("model.visual.pos_embed.", "pos_embed."),
        # vision transformer blocks
        ("model.visual.blocks.", "blocks."),
        # attention
        (".attn.qkv.", ".att.qkv."),
        (".attn.proj.", ".att.proj."),
        # FFN
        (".mlp.linear_fc1.", ".ffn.lin1."),
        (".mlp.linear_fc2.", ".ffn.lin2."),
        # norm1 and norm2 already match between HF and ours
        # merger / merge_adapter
        ("model.visual.merger.norm.", "merge_adapter.norm."),
        ("model.visual.merger.linear_fc1.", "merge_adapter.lin1."),
        ("model.visual.merger.linear_fc2.", "merge_adapter.lin2."),
    ]
    return rules


def load_qwen3_5_text_weights(model, model_cfg):
    """
    Download Qwen3.5 weights from HF and convert + load to our implementation.

    returns:
        model: our Qwen3.5 model with corresponding weights loaded
    """
    hf_model_name = model_cfg["model_path"]

    #########################
    ### Download weights ###
    hf_state_dict = download_hf_weights(hf_model_name)

    # Get remapping rules and our model state dict
    remapping_rules = get_remapping_rules()
    our_state_dict = model.state_dict()

    ########################
    ### Convert weights ###
    print("\nConverting weights...")
    converted_weights = convert_weights(
        hf_state_dict, our_state_dict, remapping_rules, ignored_prefixes=("model.visual.", "mtp.")
    )

    #####################
    ### Load weights ###
    with torch.no_grad():
        load_result = model.load_state_dict(converted_weights, strict=False)
        handle_weight_tying(model)

    report_loading_status(model, load_result, converted_weights)

    return model


def load_qwen3_5_vlm_weights(model, model_cfg):
    """
    Download Qwen3.5 weights from HF and convert + load to our VLM implementation.
    Loads both text AND vision weights.

    returns:
        model: our Qwen3.5 VLM model with corresponding weights loaded
    """
    hf_model_name = model_cfg["model_path"]

    #########################
    ### Download weights ###
    hf_state_dict = download_hf_weights(hf_model_name)

    ########################
    ### Convert weights ###
    print("\nConverting weights (text + vision)...")

    # Convert text weights separately
    text_state_dict = model.language_model.state_dict()
    text_converted = convert_weights(
        hf_state_dict, text_state_dict, get_remapping_rules(), ignored_prefixes=("model.visual.", "mtp.")
    )

    # Convert vision weights separately
    vision_state_dict = model.vision_model.state_dict()
    vision_converted = convert_weights(
        hf_state_dict,
        vision_state_dict,
        get_vision_remapping_rules(),
        ignored_prefixes=("model.language_model.", "mtp."),
    )

    #####################
    ### Load weights ###
    with torch.no_grad():
        load_result_text = model.language_model.load_state_dict(text_converted, strict=False)
        load_result_vision = model.vision_model.load_state_dict(vision_converted, strict=False)
        handle_weight_tying(model.language_model)

    # Local reconstruct load results for uniform reporting (overlaps)
    from collections import namedtuple

    LoadResult = namedtuple("LoadResult", ["missing_keys", "unexpected_keys"])

    missing_keys = [f"language_model.{k}" for k in load_result_text.missing_keys] + [
        f"vision_model.{k}" for k in load_result_vision.missing_keys
    ]
    unexpected = [f"language_model.{k}" for k in load_result_text.unexpected_keys] + [
        f"vision_model.{k}" for k in load_result_vision.unexpected_keys
    ]

    combined_result = LoadResult(missing_keys, unexpected)

    combined_weights = {f"language_model.{k}": v for k, v in text_converted.items()}
    combined_weights.update({f"vision_model.{k}": v for k, v in vision_converted.items()})

    report_loading_status(model, combined_result, combined_weights)

    return model


if __name__ == "__main__":
    torch.manual_seed(123)

    model_cfg = QWEN3_5_08B_CONFIG
    model_text = Qwen3_5TextModel(model_cfg)
    model_text = load_qwen3_5_text_weights(model=model_text, model_cfg=model_cfg)
    test_generation_with_weights(model_text, model_cfg)

    # test Final VLM weights load correctly
    print("\n------\n")
    model_cfg = QWEN3_5_08B_CONFIG
    model_vlm = Qwen3_5VLM(model_cfg)
    model_vlm = load_qwen3_5_vlm_weights(model=model_vlm, model_cfg=model_cfg)
