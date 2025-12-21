from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

import config
from llm_quest.alignment.rlvr_grpo_reasoning.rlvr_engine import rlvr_grpo_prompt_collator, rlvr_grpo_training_loop
from llm_quest.dataset import RPTStructuredDataset
from llm_quest.engine import LearningRateScheduler
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model
from llm_quest.qwen.qwen3.qwen3_weight_loading import load_qwen3_weights
from llm_quest.reinforcement_pretraining.rpt_engine import PrefixMatchingReward

# reasoning model triggered by chat template
policy_cfg = config.qwen3_config_creator("0.6B", base_model=False)
reference_cfg = policy_cfg.copy()

tokenizer = AutoTokenizer.from_pretrained(policy_cfg["model_path"])
pad_token_id = tokenizer.pad_token_id
eos_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
model_device = config.auto_device

# --- hyperparameters ---
batch_size = 2  # prompts batch size, responses batch size will be batch_size * num_samples(rollout)
labels_length = 3
min_context_tokens = 15  # avoids artificially hard samples to predict without enough context
truncate_sample = (0, 100)

# optimizer and LR scheduler hparams
weight_decay = 0.1
peak_lr = 1e-6
init_lr = 0.0
warmup_steps = 18
min_lr = None
decay = None

# loader hparams
num_workers = 0
pin_memory = False
persistent_workers = False

# Training
use_gradient_checkpointing = False  # trade memory for speed if disabled
rpt_training_hparams = {
    "num_epoch": 1,
    "max_gen": 700,
    "eos_ids": eos_token_ids,
    "pad_id": pad_token_id,
    "sampling_params": {  # recommended values by Qwen for non-base models
        "top_k": 20,
        "top_p": 0.95,
        "min_p": None,
        "temp": 0.6,
    },
    # GRPO
    "loss_variant": "grpo",  # alt: dapo, dr_grpo, gspo, sapo
    "num_samples": 2,
    "num_grad_updates": 2,
    "min_clip_eps": 0.2,
    "max_clip_eps": 0.2,
    "beta": 0.0,
    # eval
    "evaluation": True,
    "eval_freq": 10,
    "eval_batches": 1,
    "eval_num_samples": 2,
    # thresholds for checkpoint saving
    "kl_div_threshold": 0.3,
    "min_reward_threshold": 0.35,
}

# Formulating some instructions... (sensitive task)
instruction = (
    "Complete the only given text under '### Text' by predicting the next word. "
    "There is no more context besides the given text under '### Text'.\n"
    "Instructions: "
    "1. List exactly 2 candidate words. Do not provide any explanation or reasoning beyond the word list. "
    "2. Immediately select the most probable word and enclose it in <answer> </answer> tags. "
    "Do not write any other text or explanation. "
    "(note: the word may begin with a space, e.g., '<answer> para</answer>'.\n\n"
    "### Text\n"
)


if __name__ == "__main__":

    config.use_phantom_reward = True  # overriding config to use phantom reward specifically for RPT
    torch.manual_seed(123)

    reward_calculator = PrefixMatchingReward(tokenizer=tokenizer, good_answer_reward=2.0, dtype=policy_cfg["dtype"])

    # --- datasets & loaders ---
    # reusing GSM8K dataset instead of Omni-Math
    # for targeted efficient training: filter the dataset beforehand to include only predictions of certain difficulty
    # via the EntropyFilteredTokens class
    train_set = RPTStructuredDataset(
        config.reasoning_train_path,
        instruction=instruction,
        tokenizer=tokenizer,
        max_context_length=policy_cfg["context_length"],
        labels_length=labels_length,
        apply_chat_template=True,
        truncate_sample=truncate_sample,
        min_context_tokens=min_context_tokens,
    )
    val_set = RPTStructuredDataset(
        config.reasoning_val_path,
        instruction=instruction,
        tokenizer=tokenizer,
        max_context_length=policy_cfg["context_length"],
        labels_length=labels_length,
        apply_chat_template=True,
        truncate_sample=truncate_sample,
        min_context_tokens=min_context_tokens,
    )


train_set = Subset(train_set, range(500))
val_set = Subset(val_set, range(50))


# same as RLVR
custom_collate = partial(
    rlvr_grpo_prompt_collator,
    custom_max_length=policy_cfg["context_length"],
    device=model_device,
    pad_token_id=pad_token_id,
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
# note: the rpt_grpo training loop will take care of putting models on correct training/eval mode
policy_model = Qwen3Model(policy_cfg)
reference_model = Qwen3Model(reference_cfg)
policy_model = load_qwen3_weights(policy_model, policy_cfg)
policy_model.gradient_checkpointing = use_gradient_checkpointing

policy_model.to(device=model_device, dtype=torch.bfloat16)
reference_model.to(device=model_device, dtype=torch.bfloat16).eval()

# no need to set optimizer's lr, the LR scheduler will init optimizer's lr
optimizer = torch.optim.AdamW(policy_model.parameters(), weight_decay=weight_decay, fused=True)

total_steps = len(train_loader) * rpt_training_hparams["num_epoch"]
lr_scheduler = LearningRateScheduler(
    optimizer,
    total_steps=total_steps,
    init_lr=init_lr,
    peak_lr=peak_lr,
    warmup_steps=warmup_steps,
    min_lr=min_lr,
    decay=decay,
)

rlvr_grpo_training_loop(
    train_loader=train_loader,
    val_loader=val_loader,
    policy_model=policy_model,
    reference_model=reference_model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    reward_calculator=reward_calculator,
    policy_config=policy_cfg,
    device=model_device,
    **rpt_training_hparams,
)
