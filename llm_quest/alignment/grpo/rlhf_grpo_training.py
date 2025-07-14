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
gpt_config = config.config_creator("gpt_m")
model_device = "cuda"
# optimizer hparams (alt hparams in comments: slower, more stable, learning)
lr = 5e-5  # alt 3e-5
weight_decay = 0.01
# training hparams
batch_size = 4  # alt 8
num_samples = 6  # alt 4
num_epoch = 1  # alt 2
num_grad_updates = 3  # alt 1 or 2
max_gen = 35
# GRPO hparams
eps = 0.2  # alt 0.15
beta = 0.2  # alt 0.1
# evaluation hparams
evaluation = True
eval_freq = 10  # alt 20
eval_batches = 1  # alt 2
eval_num_samples = 5  # alt 4
kl_div_threshold = 0.75  # alt 0.5
# loader hparams
num_workers = 0
pin_memory = False
persistent_workers = False


if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- datasets & loaders ---
    train_set = PreferenceDataset(config.instruct_preference_train_path, tokenizer, prompts_only=True)
    val_set = PreferenceDataset(config.instruct_preference_val_path, tokenizer, prompts_only=True)

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

    # --- models initialization & loading ---
    # note: the grpo training loop will take care of putting models on correct training/eval mode
    policy_model = GPTModel(gpt_config)
    reference_model = GPTModel(gpt_config)
    reward_model = GPTModel(gpt_config)
    reward_model.out = nn.Linear(gpt_config["emb_dim"], 1)

    pol_checkpoint = torch.load(config.ft_instruct_w_gpt2, map_location="cpu", weights_only=True)
    reward_checkpoint = torch.load(config.reward_model_pref_tuning, map_location="cpu", weights_only=True)
    policy_model.load_state_dict(pol_checkpoint["model_state_dict"])
    reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
    del pol_checkpoint, reward_checkpoint  # removing upfront rather than waiting for gc to kick in

    reward_model.to(device=model_device, dtype=torch.bfloat16)
    policy_model.to(device=model_device, dtype=torch.bfloat16)
    reference_model.to(device=model_device, dtype=torch.bfloat16)

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
        kl_div_threshold=kl_div_threshold,
    )
