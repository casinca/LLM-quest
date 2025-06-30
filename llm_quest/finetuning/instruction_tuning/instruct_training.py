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
warmup_percent = 0.1
eval_freq = 100
eval_iter = 10
weight_decay = 0.1
accumulation_steps = 2
num_workers = 0
pin_memory = False
use_amp = False
model_cfg = config.config_creator("gpt_m")

data_device = "cpu"
model_device = "cuda"

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers

    from gpt_download import download_and_load_gpt2
    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import training_eval_loop
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import load_weights_into_gpt

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Loaders ---
    settings, params = download_and_load_gpt2(model_size="355M", models_dir=config.openai_pretrained_w_gpt2_m)
    model = GPTModel(model_cfg)
    load_weights_into_gpt(model, params)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay, fused=True)

    training_eval_loop(
        train_loader,
        val_loader,
        model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        warmup_percent=warmup_percent,
        init_lr=init_lr,
        peak_lr=peak_lr,
        min_lr=min_lr,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        device=model_device,
        accumulation_steps=accumulation_steps,
        use_amp=use_amp,
    )

    # save instruct finetuned model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
        },
        config.ft_instruct_w_gpt2,
    )
