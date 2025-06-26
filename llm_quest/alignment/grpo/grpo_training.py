from functools import partial

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.grpo.grpo_engine import grpo_prompt_collator, grpo_training_loop
from llm_quest.dataset import PreferenceDataset
from llm_quest.gpt.gpt_model import GPTModel

# --- hyperparameters ---
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
gpt_config = config.GPT_SMALL_CONFIG
model_device = "cuda"
# optimizer hparams
lr = 1e-5
weight_decay = 0.01
# training hparams
batch_size = 4
num_samples = 5
num_epoch = 1
num_grad_updates = 4
max_gen = 35
# GRPO hparams
eps = 0.2
beta = 1.0
# evaluation hparams
eval_freq = 5
evaluation = True
eval_batches = 1
eval_num_samples = 1
# loader hparams
num_workers = 0
pin_memory = False
persistent_workers = False


if __name__ == "__main__":

    # --- datasets & loaders ---
    train_set = PreferenceDataset(config.instruct_preference_train_path, tokenizer)
    val_set = PreferenceDataset(config.instruct_preference_val_path, tokenizer)

    custom_collate = partial(
        grpo_prompt_collator,
        custom_max_length=gpt_config["context_length"],
        device=model_device,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # --- models & optimizer ---
    # note: the grpo training loop will take care of putting models on correct training/eval mode
    policy_model = GPTModel(gpt_config)
    reference_model = GPTModel(gpt_config)
    reward_model = GPTModel(gpt_config)
    reward_model.out = nn.Linear(gpt_config["emb_dim"], 1)  # (testing untrained, otherwise no need to do this)

    reward_model.to(model_device)
    policy_model.to(model_device)
    reference_model.to(model_device)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)

    grpo_training_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        num_samples=num_samples,
        num_grad_updates=num_grad_updates,
        policy_config=gpt_config,
        device=model_device,
        max_gen=max_gen,
        eps=eps,
        beta=beta,
        evaluation=evaluation,
        eval_freq=eval_freq,
        eval_batches=eval_batches,
        eval_num_samples=eval_num_samples,
    )
