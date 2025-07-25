from pathlib import Path

import torch

# ----------- OG CONFIGS -----------


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,  # Query-Key-Value bias
}


LLAMA32_SMALL_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "num_kv_groups": 8,
    "rope_base": 500_000,
    "rope_freq": {
        "factor": 32.0,
        "alpha": 1.0,
        "beta": 4.0,
        "original_context_length": 8192,
    },
    # brain f16, less precision than f16 but better range (similar to f32)
    # (sacrificing precision for less over/underflow issues)
    "dtype": torch.bfloat16,
}

# ----------- CUSTOM CONFIGS -----------

GPT_SMALL_CONFIG = {
    "vocab_size": 50304,  # changed from 50257 to closest multiple of 64
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0,
    "qkv_bias": False,
}


LLAMA32_SMALL_CONFIG = {
    "vocab_size": 50304,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "num_kv_groups": 4,
    "hidden_dim": 4 * 768,  # equivalent to GPT hidden dim
    "rope_base": 10_000,  # RoPE base for θ calc
    # this section is mainly for SFT/Context length extension
    "rope_freq": {  # hparams for RoPE variant (NTK Aware + by parts/wavelength scaling)
        "factor": 32.0,  # or L'/L
        "alpha": 1.0,  # low frequency boundary
        "beta": 32.0,  # high frequency boundary
        "og_ctx_len": 4096,  # the original context length L
        "ctx_len": 8192,  # the new context length L'
    },
    "dtype": torch.float32,
}


GEMMA3_SMALL_CONFIG = {
    "vocab_size": 50304,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "num_kv_groups": 6,
    "hidden_dim": 4 * 768,
    "window_size": 3,
    "local_global_att_ratio": 5,  # O = full global & 'n_layers' = full SWA
    "rope_base": 10_000,
    "rope_freq": {
        "factor": 32.0,
        "alpha": 1.0,
        "beta": 32.0,
        "og_ctx_len": 4096,
        "ctx_len": 8192,
    },
    "dtype": torch.float32,
}

# TODO remove hardcoded values for DeepSeekMoE and add it to the config

DEEPSEEK_SMALL_CONFIG = {
    "vocab_size": 50304,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "hidden_dim": 4 * 768,
    "num_ffn": 3,  # number of FFN layers, remaining will be MoE
    "mtp_depth": 2,  # number of MTP modules (depth of the multi token prediction)
    "mtp_loss_coeff": 0.2,  # for now static, DeepSeek mentions 0.1 for first 10T tokens, 0.3 for the remaining 4.8T
    "rope_base": 10_000,
    "rope_freq": {
        "factor": 32.0,
        "alpha": 1.0,
        "beta": 32.0,
        "og_ctx_len": 4096,
        "ctx_len": 8192,
    },
    "dtype": torch.float32,
}

# ViT config (similar to the ones in the paper)
VIT_BASE_CONFIG = {
    "img_width": 224,
    "img_height": 224,
    "patch_size": 16,
    "num_channels": 3,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "num_classes": 100,
}

TINY_VIT_CONFIG = {
    "img_width": 32,
    "img_height": 32,
    "patch_size": 4,  # Smaller patches for 32x32 images (8 patches total)
    "num_channels": 3,  # RGB
    "emb_dim": 256,
    "n_layers": 12,
    "n_heads": 8,
    "drop_rate": 0.3,
    "qkv_bias": True,
    "num_classes": 10,  # CIFAR-10 has... 10 classes
}


def config_creator(gpt_size):
    """
    This function creates a config dictionary for a GPT model based on the size provided.

    Args:
        gpt_size (str): The size of the gpt model. It can be one of the following:
            "gpt_s"  → gpt2-small (124M)\n
            "gpt_m"  → gpt2-medium (355M)\n
            "gpt_l"  → gpt2-large (774M)\n
            "gpt_xl" → gpt2-xl (1558M)

    Returns:
        dict: A dictionary containing the configuration parameters for the gpt model.
    """

    model_configs = {
        "gpt_s": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "qkv_bias": True},
        "gpt_m": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "qkv_bias": True},
        "gpt_l": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "qkv_bias": True},
        "gpt_xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "qkv_bias": True},
    }

    new_config = GPT_CONFIG_124M.copy()
    new_config.update(model_configs[gpt_size])

    return new_config


# ----------- PATHS -----------

# Get the relative path (of the config file, at root) convert to absolute directory
root_dir = Path(__file__).resolve().parent

# --- datasets ---
the_verdict_path = root_dir / "data" / "the-verdict.txt"

spam_train_path = root_dir / "data" / "spam_processed" / "train.csv"
spam_val_path = root_dir / "data" / "spam_processed" / "validation.csv"
spam_test_path = root_dir / "data" / "spam_processed" / "test.csv"

instruct_train_path = root_dir / "data" / "instruction_processed" / "train_set.json"
instruct_val_path = root_dir / "data" / "instruction_processed" / "val_set.json"
instruct_test_path = root_dir / "data" / "instruction_processed" / "test_set.json"

instruct_alpaca_train_path = root_dir / "data" / "instruct_alpaca" / "train_set.json"
instruct_alpaca_val_path = root_dir / "data" / "instruct_alpaca" / "val_set.json"
instruct_alpaca_test_path = root_dir / "data" / "instruct_alpaca" / "test_set.json"

instruct_preference_train_path = root_dir / "data" / "instruct_preference_processed" / "train_set.json"
instruct_preference_val_path = root_dir / "data" / "instruct_preference_processed" / "val_set.json"
instruct_preference_test_path = root_dir / "data" / "instruct_preference_processed" / "test_set.json"

fineweb_train = root_dir / "data" / "fineweb_sample" / "train_fineweb.jsonl.gz"
fineweb_val = root_dir / "data" / "fineweb_sample" / "val_fineweb.jsonl.gz"

reasoning_train_path = root_dir / "data" / "processed_data" / "gsm8k_processed" / "gsm8k_train.jsonl"
reasoning_val_path = root_dir / "data" / "processed_data" / "gsm8k_processed" / "gsm8k_test.jsonl"

# --- models ---
openai_pretrained_w_gpt2 = root_dir / "checkpoints" / "gpt2"
openai_pretrained_w_gpt2_m = root_dir / "checkpoints" / "gpt2_medium"
custom_pretrained_w_gpt2 = root_dir / "checkpoints" / "model_and_optim_save.pth"

ft_classifier_w_gpt2 = root_dir / "checkpoints" / "ft_classifier_model_and_optim_save.pth"

ft_instruct_w_gpt2 = root_dir / "checkpoints" / "ft_instruct_model_and_optim_save.pth"

reward_model_pref_tuning = root_dir / "checkpoints" / "reward_model_pref_tuning.pth"
grpo_policy_model = root_dir / "checkpoints" / "grpo_policy_model.pth"
rlhf_grpo_checkpoint_dir = root_dir / "checkpoints" / "rlhf_grpo_checkpoints"
rlhf_rm_checkpoint_dir = root_dir / "checkpoints" / "rlhf_rm_checkpoints"

rlvr_grpo_checkpoint_dir = root_dir / "checkpoints" / "rlvr_grpo_checkpoints"

sft_reasoning_gpt2 = root_dir / "checkpoints" / "sft_reasoning_model_save_amp.pth"
