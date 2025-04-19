from functools import partial

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.grpo.grpo_engine import bt_loss, reward_model_training_eval_loop_simple
from llm_quest.dataset import PreferenceDataset, custom_collate_fn
from llm_quest.gpt.gpt_model import GPTModel

# --- hyperparameters ---
torch.manual_seed(42)
batch_size = 8
lr = 1e-5
weight_decay = 0.01
num_epoch = 1
data_device = "cpu"
model_device = "cuda"
model_cfg = config.GPT_SMALL_CONFIG
tokenizer = tiktoken.get_encoding("gpt2")

# --- datasets ---
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

# reward_model_training_eval_loop_simple(
#    train_loader,
#    val_loader,
#    model,
#    optimizer,
#    num_epoch,
#    eval_freq=10,
# )

# --- training loop ---
test_batch = next(iter(train_loader))
print("test_batch['chosen'].shape", test_batch["chosen"].shape)

pref_mini_rewards = model(test_batch["chosen"]).squeeze(-1)
reject_mini_rewards = model(test_batch["rejected"]).squeeze(-1)

print("pref_mini_rewards.shape", pref_mini_rewards.shape)

pref_mask = test_batch["chosen_mask"]  # shape (b, s) -> (b, s, 1)
reject_mask = test_batch["rejected_mask"]
pref_mini_rewards *= pref_mask
reject_mini_rewards *= reject_mask
# --- mean pooling over the sequence length ---
num_valid_pref_tokens = pref_mask.sum(dim=1)  # we want to divide by the number of valid tokens
num_valid_reject_tokens = reject_mask.sum(dim=1)
pref_rewards = pref_mini_rewards.sum(dim=1) / num_valid_pref_tokens
reject_rewards = reject_mini_rewards.sum(dim=1) / num_valid_reject_tokens

print("pref_rewards.shape", pref_rewards.shape)
print("pref_rewards", pref_rewards)
# shape (b,1) -> (b)
pref_rewards, reject_rewards = pref_rewards, reject_rewards

loss = bt_loss(pref_rewards, reject_rewards)

print(loss)
