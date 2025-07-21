from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.rlhf_grpo.grpo_engine import reward_model_training_eval_loop_simple
from llm_quest.alignment.rlhf_grpo.pref_reward_model import PreferenceRewardModel
from llm_quest.dataset import PreferenceDataset, pref_reward_collate

# --- hyperparameters ---
batch_size = 8
lr = 6e-5
weight_decay = 0.1
num_epoch = 2
beta = 1.0
data_device = "cpu"
model_device = "cuda"
model_cfg = config.config_creator("gpt_m")
model_cfg["drop_rate"] = 0.2
eval_freq = 10
eval_num_batches = 8
num_workers = 0
pin_memory = False

if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- datasets & loaders ---
    train_set = PreferenceDataset(config.instruct_preference_train_path, tokenizer)
    val_set = PreferenceDataset(config.instruct_preference_val_path, tokenizer)

    pref_rm_collate = partial(
        pref_reward_collate,
        allowed_max_length=model_cfg["context_length"],
        device=model_device,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=pref_rm_collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=pref_rm_collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- model & optimizer ---
    model = PreferenceRewardModel(model_cfg)
    checkpoint = torch.load(config.ft_instruct_w_gpt2, weights_only=True, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("out.weight")  # removing SFT head weights, we want to train a new scalar head
    model.load_state_dict(state_dict, strict=False)  # strict=False to ignore the missing layer key
    del checkpoint  # removing upfront rather than waiting for gc to kick in

    model.to(device=model_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)

    reward_model_training_eval_loop_simple(
        train_loader,
        val_loader,
        model,
        optimizer,
        num_epoch,
        eval_freq=eval_freq,
        eval_num_batches=eval_num_batches,
        beta=beta,
    )
