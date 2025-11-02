from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

torch.manual_seed(123)

# --- Hyperparameters ---
batch_size = 4
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
    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import training_eval_loop
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import alpaca_deepseek_format

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
        config.sft_reasoning_gpt2,
    )
