from functools import partial

import torch
import transformers
from torch.utils.data import DataLoader

import config
from llm_quest.alignment.rlvr_grpo_reasoning.rlvr_engine import rlvr_grpo_prompt_collator, rlvr_grpo_training_loop
from llm_quest.dataset import ReasoningDataset
from llm_quest.gpt.gpt_model import GPTModel

# --- hyperparameters ---
gpt_config = config.config_creator("gpt_m")
model_device = "cuda"
# optimizer hparams
lr = 5e-5
weight_decay = 0.1
# training hparams
batch_size = 4
num_samples = 4
num_epoch = 1
num_grad_updates = 3
max_gen = 250
# GRPO hparams
loss_variant = "grpo"
clip_eps = 0.2
beta = 0.45
# evaluation hparams
evaluation = True
eval_freq = 50
eval_batches = 1
eval_num_samples = 4
kl_div_threshold = 0.3
# loader hparams
num_workers = 0
pin_memory = False
persistent_workers = False


if __name__ == "__main__":
    torch.manual_seed(123)

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")  # using HF tokenizer mainly for batch_decode

    # --- datasets & loaders ---
    train_set = ReasoningDataset(config.reasoning_train_path, tokenizer)
    val_set = ReasoningDataset(config.reasoning_val_path, tokenizer)

    custom_collate = partial(
        rlvr_grpo_prompt_collator,
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
    # note: the rlvr_grpo training loop will take care of putting models on correct training/eval mode
    policy_model = GPTModel(gpt_config)
    reference_model = GPTModel(gpt_config)

    pol_checkpoint = torch.load(config.sft_reasoning_gpt2, map_location="cpu", weights_only=True)
    policy_model.load_state_dict(pol_checkpoint["model_state_dict"])
    del pol_checkpoint  # removing upfront rather than waiting for gc to kick in

    policy_model.to(device=model_device, dtype=torch.bfloat16)
    reference_model.to(device=model_device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)

    rlvr_grpo_training_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        num_epoch=num_epoch,
        num_samples=num_samples,
        num_grad_updates=num_grad_updates,
        policy_config=gpt_config,
        device=model_device,
        max_gen=max_gen,
        clip_eps=clip_eps,
        beta=beta,
        evaluation=evaluation,
        eval_freq=eval_freq,
        eval_batches=eval_batches,
        eval_num_samples=eval_num_samples,
        kl_div_threshold=kl_div_threshold,
        loss_variant=loss_variant,
    )
