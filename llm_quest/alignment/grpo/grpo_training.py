from functools import partial

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.grpo.grpo_engine import grpo_training_loop
from llm_quest.dataset import PreferenceDataset, custom_collate_fn
from llm_quest.gpt.gpt_model import GPTModel

# --- hyperparameters ---
torch.manual_seed(123)
gpt_config = config.GPT_SMALL_CONFIG
batch_size = 8
lr = 1e-5
weight_decay = 0.01
num_epoch = 1
num_samples = 3
num_grad_updates = 2
tokenizer = tiktoken.get_encoding("gpt2")
model_device = "cuda"


# --- datasets ---
test_set = PreferenceDataset(config.instruct_preference_test_path, tokenizer)
custom_collate = partial(
    custom_collate_fn,
    allowed_max_length=gpt_config["context_length"],
    mask_prompt_tokens=True,
    device=model_device,
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    collate_fn=custom_collate,
    shuffle=False,
)

# --- models & optimizer ---
# note: the grpo training loop will take care of putting models on correct training/eval mode
# note2: copying policy_model's parameters to the reference_model is also done in the grpo training loop
policy_model = GPTModel(gpt_config)
reference_model = GPTModel(gpt_config)
reward_model = GPTModel(gpt_config)
reward_model.out = nn.Linear(gpt_config["emb_dim"], 1)

reward_model.to(model_device)
policy_model.to(model_device)
reference_model.to(model_device)

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)

grpo_training_loop(
    train_loader=test_loader,
    policy_model=policy_model,
    reference_model=reference_model,
    reward_model=reward_model,
    optimizer=optimizer,
    num_epoch=num_epoch,
    num_samples=num_samples,
    num_grad_updates=num_grad_updates,
    policy_config=gpt_config,
    device=model_device,
    eps=0.2,
    beta=1.0,
)
