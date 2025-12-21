from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from llm_quest.alignment.rlvr_grpo_reasoning.rlvr_engine import (
    VerifiableRewardCalculator,
    rlvr_grpo_prompt_collator,
    rlvr_grpo_training_loop,
)
from llm_quest.dataset import ReasoningDataset
from llm_quest.engine import LearningRateScheduler
from llm_quest.gpt.gpt_model import GPTModel

gpt_config = config.gpt2_config_creator("gpt_m")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.pad_token_id == tokenizer.eos_token_id == 50256
eos_and_pad_id = tokenizer.eos_token_id

model_device = config.auto_device

# --- hyperparameters ---
batch_size = 2  # prompts batch size, responses batch size will be batch_size * num_samples(rollout)

# optimizer hparams
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

# training
rlvr_training_hparams = {
    "num_epoch": 1,
    "max_gen": 250,
    "eos_ids": eos_and_pad_id,
    "pad_id": eos_and_pad_id,
    "sampling_params": {
        "top_k": 20,
        "top_p": None,
        "min_p": None,
        "temp": 1.0,
    },
    # GRPO hparams
    "loss_variant": "grpo",
    "num_samples": 2,
    "num_grad_updates": 2,
    "min_clip_eps": 0.2,
    "max_clip_eps": 0.2,
    "beta": 0.45,
    # evaluation hparams
    "evaluation": True,
    "eval_freq": 50,
    "eval_batches": 1,
    "eval_num_samples": 4,
    # thresholds for checkpoint saving
    "kl_div_threshold": 0.3,
    "min_reward_threshold": 0.35,
}


if __name__ == "__main__":
    torch.manual_seed(123)
    config.use_phantom_reward = True
    reward_calculator = VerifiableRewardCalculator(tokenizer=tokenizer)

    # --- datasets & loaders ---
    train_set = ReasoningDataset(config.reasoning_train_path, tokenizer)
    val_set = ReasoningDataset(config.reasoning_val_path, tokenizer)

    custom_collate = partial(
        rlvr_grpo_prompt_collator,
        custom_max_length=gpt_config["context_length"],
        device=model_device,
        pad_token_id=eos_and_pad_id,
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
    # note: the rlvr_grpo training loop will take care of putting models on correct training/eval mode
    policy_model = GPTModel(gpt_config)
    reference_model = GPTModel(gpt_config)

    pol_checkpoint = torch.load(config.sft_reasoning_gpt2, map_location="cpu", weights_only=True)
    policy_model.load_state_dict(pol_checkpoint["model_state_dict"])
    del pol_checkpoint  # removing upfront rather than waiting for gc to kick in

    policy_model.to(device=model_device, dtype=torch.bfloat16)
    reference_model.to(device=model_device, dtype=torch.bfloat16)

    # no need to set optimizer's lr, the LR scheduler will init optimizer's lr
    optimizer = torch.optim.AdamW(policy_model.parameters(), weight_decay=weight_decay, fused=True)

    total_steps = len(train_loader) * rlvr_training_hparams["num_epoch"]
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
        policy_config=gpt_config,
        device=model_device,
        reward_calculator=reward_calculator,
        **rlvr_training_hparams,
    )
