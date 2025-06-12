from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

torch.manual_seed(123)

# --- Hyperparameters ---
batch_size = 8
num_epoch = 2
peak_lr = 1e-5
warmup_percent = 0
init_lr = 1e-5
min_lr = 1e-5
eval_freq = 100
eval_iter = 10
weight_decay = 0.1
accumulation_steps = 2
num_workers = 0
pin_memory = False
use_amp = True

data_device = "cpu"
model_device = "cuda"

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers
    import config
    from gpt_download import download_and_load_gpt2
    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import training_eval_loop
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import load_weights_into_gpt

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Loaders ---
    model_cfg = config.config_creator("gpt_m")  # using Medium sized gpt config
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
    model.to(model_device)

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
            "optimizer_state_dict": optimizer.state_dict(),
        },
        config.ft_instruct_w_gpt2,
    )
