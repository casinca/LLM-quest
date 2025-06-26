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
torch.manual_seed(123)
batch_size = 8
lr = 1e-5
weight_decay = 0.01
num_epoch = 1
data_device = "cpu"
model_device = "cuda"
model_cfg = config.GPT_SMALL_CONFIG
tokenizer = tiktoken.get_encoding("gpt2")
eval_freq = 10

if __name__ == "__main__":

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
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=dpo_custom_collate,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    # --- model & optimizer ---
    model = GPTModel(model_cfg)
    # changing the head to a single output linear layer: we want a scalar reward
    model.out = nn.Linear(model_cfg["emb_dim"], 1)
    model.to(model_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    reward_model_training_eval_loop_simple(
        train_loader,
        val_loader,
        model,
        optimizer,
        num_epoch,
        eval_freq=eval_freq,
    )

    torch.save(model.state_dict(), config.reward_model_pref_tuning)
