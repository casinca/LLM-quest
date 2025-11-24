from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

torch.manual_seed(123)

# --- Hyperparameters ---
batch_size = 4
num_epoch = 2
peak_lr = 1e-5
warmup_steps = 0
init_lr = 1e-5
min_lr = None
decay = None
eval_freq = 100
eval_iter = 10
weight_decay = 0.1
accumulation_steps = 2
num_workers = 0
pin_memory = False
use_amp = True

data_device = "cpu"

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers
    import math

    import config
    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import LearningRateScheduler, training_eval_loop
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import alpaca_deepseek_format

    model_device = config.auto_device
    # torch.set_float32_matmul_precision("high")
    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Loaders ---
    model_cfg = config.gpt2_config_creator("gpt_m")

    model = GPTModel(model_cfg)
    checkpoint = torch.load(config.ft_instruct_w_gpt2, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint

    train_set = InstructionDataset(
        "../../../data/processed_data/gsm8k_processed/gsm8k_train.jsonl",
        tokenizer,
        formatting_func=alpaca_deepseek_format,
        file_type="jsonl",
    )[:1000]

    val_set = InstructionDataset(
        "../../../data/processed_data/gsm8k_processed/gsm8k_train.jsonl",
        tokenizer,
        formatting_func=alpaca_deepseek_format,
        file_type="jsonl",
    )[1000:1101]

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

    model.to(model_device)

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

    train_losses, val_losses = training_eval_loop(
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
    #    config.sft_reasoning_gpt2,
    # )
