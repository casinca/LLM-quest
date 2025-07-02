from functools import partial

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.grpo.grpo_engine import reward_model_training_eval_loop_simple
from llm_quest.dataset import PreferenceDataset, custom_collate_fn
from llm_quest.gpt.gpt_model import GPTModel

# --- hyperparameters ---
batch_size = 8
lr = 1e-4
weight_decay = 0.01
num_epoch = 2
beta = 1.0
data_device = "cpu"
model_device = "cuda"
model_cfg = config.config_creator("gpt_m")
model_cfg["drop_rate"] = 0.1
eval_freq = 10
eval_num_batches = 5
num_workers = 0
pin_memory = False

if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- datasets & loaders ---
    train_set = PreferenceDataset(config.instruct_preference_train_path, tokenizer)
    val_set = PreferenceDataset(config.instruct_preference_val_path, tokenizer)

    dpo_custom_collate = partial(
        custom_collate_fn,
        allowed_max_length=model_cfg["context_length"],
        mask_prompt_tokens=True,
        device=model_device,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=dpo_custom_collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=dpo_custom_collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- model & optimizer ---
    model = GPTModel(model_cfg)
    checkpoint = torch.load(config.ft_instruct_w_gpt2, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint  # removing upfront rather than waiting for gc to kick in

    # changing the head to a single output linear layer: we want a scalar reward
    model.out = nn.Linear(model_cfg["emb_dim"], 1)
    model.to(device=model_device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    torch.save(
        {"model_state_dict": model.state_dict()},
        config.reward_model_pref_tuning,
    )
