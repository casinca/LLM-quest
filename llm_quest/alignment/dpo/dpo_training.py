from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.dpo.dpo import dpo_training_eval_loop_simple
from llm_quest.dataset import PreferenceDataset, custom_collate_fn
from llm_quest.gpt.gpt_model import GPTModel

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch_size = 8

data_device = "cpu"
model_device = "cuda"

model_cfg = config.config_creator("gpt_m")
# the ref model was SFT on a pretrained GPT2 with OpenAI weights, can't use 50304.
model_cfg["vocab_size"] = 50257

# instantiating the Policy model
policy_model = GPTModel(model_cfg)
# loading our finetuned model params, weights_only=True, not to using prev optim params
ft_checkpoint = torch.load(config.ft_instruct_w_gpt2, map_location=data_device, weights_only=True)
policy_model.load_state_dict(ft_checkpoint["model_state_dict"])
policy_model.to(model_device)

# instantiating the Reference_model
reference_model = GPTModel(model_cfg)
# copy state_dict from the policy model
reference_model.load_state_dict(ft_checkpoint["model_state_dict"])
reference_model.to(model_device)
reference_model.eval()  # important to put the reference model in eval mode

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

num_epoch = 1  # 1 epoch to avoid catastrophic forgetting
# @rasbt mentioned, for dpo one should use a lower lr than pretraining or SFT.
# Thus, I'm not loading optim state dict. One reason at least, is that the nature of the loss is inherently different
# from pretraining or SFT.
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

dpo_training_eval_loop_simple(
    train_loader=train_loader,
    val_loader=val_loader,
    policy_model=policy_model,
    reference_model=reference_model,
    optimizer=optimizer,
    num_epoch=num_epoch,
    eval_freq=5,
    eval_iter=5,
    device=model_device,
    beta=0.1,
)
