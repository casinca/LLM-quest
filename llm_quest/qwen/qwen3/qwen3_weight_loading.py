"""
This module loads pretrained Qwen3 models from Hugging Face
and convert them to work with our custom Qwen3 implementation.
"""

import torch

from config import qwen3_config_creator
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model
from llm_quest.utils import (
    convert_weights,
    download_hf_weights,
    handle_weight_tying,
    report_loading_status,
    test_generation_with_weights,
)


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
    hf_state_dict = download_hf_weights(hf_model_name)

    # Get remapping rules and our model state dict
    remapping_rules = get_remapping_rules(model_cfg=model_cfg)
    our_state_dict = model.state_dict()

    ########################
    ### Convert weights ###
    print("\nConverting weights...")
    converted_weights = convert_weights(hf_state_dict, our_state_dict, remapping_rules)

    #####################
    ### Load weights ###
    with torch.no_grad():
        load_result = model.load_state_dict(converted_weights, strict=False)
        handle_weight_tying(model)

    report_loading_status(model, load_result, converted_weights)

    return model


if __name__ == "__main__":
    torch.manual_seed(123)

    model_cfg = qwen3_config_creator("0.6B", base_model=True)
    model = Qwen3Model(model_cfg)
    model = load_qwen3_weights(model=model, model_cfg=model_cfg)

    test_generation_with_weights(model, model_cfg)
