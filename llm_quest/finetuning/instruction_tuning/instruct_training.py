from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config

# --- Hyperparameters ---
batch_size = 16
num_epoch = 1
peak_lr = 3.9e-4
init_lr = 2.2e-4
min_lr = 2.7e-4
decay = "cosine"
warmup_steps = 100
eval_freq = 100
eval_iter = 10
weight_decay = 0.1
accumulation_steps = 2
num_workers = 0
pin_memory = False
use_amp = False
model_cfg = config.gpt2_config_creator("gpt_m")

data_device = "cpu"

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers
    import math

    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import LearningRateScheduler, training_eval_loop
    from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
    from llm_quest.gpt.gpt_model import GPTModel

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    model_device = config.auto_device

    # --- Loaders ---
    model = GPTModel(model_cfg)
    weights_path = download_gpt_model(gpt_size="gpt_m", save_dir=config.openai_pretrained_w_gpt2_m)
    load_gpt_weights(model, weights_path)

    train_set = InstructionDataset(config.instruct_alpaca_train_path, tokenizer)
    val_set = InstructionDataset(config.instruct_alpaca_val_path, tokenizer)

    # using partial() to hardcode device and custom_max_len args for the loader
    custom_collate = partial(
        collate_function,
        custom_max_len=model_cfg["context_length"],
        device=data_device,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=custom_collate,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- Training ---
    model.to(device=model_device, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, fused=True)

    update_steps = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = update_steps * num_epoch
    lr_scheduler = LearningRateScheduler(
        optimizer,
        total_steps=total_steps,
        init_lr=init_lr,
        peak_lr=peak_lr,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        decay=decay,
    )

    training_eval_loop(
        train_loader,
        val_loader,
        model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        lr_scheduler=lr_scheduler,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        device=model_device,
        accumulation_steps=accumulation_steps,
        use_amp=use_amp,
    )

    ## save instruct finetuned model
    # torch.save(
    #    {
    #        "model_state_dict": model.state_dict(),
    #        # "optimizer_state_dict": optimizer.state_dict(),
    #    },
    #    config.ft_instruct_w_gpt2,
    # )
