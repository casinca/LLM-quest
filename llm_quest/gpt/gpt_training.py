import tiktoken
import torch

torch.manual_seed(123)

# --- Hyperparameters ---
batch_size = 2
num_epoch = 10
peak_lr = 5e-4
warmup_percent = 0.2
init_lr = 1e-5
min_lr = 1e-5
eval_freq = 5
eval_iter = 5
weight_decay = 0.1
num_workers = 0
pin_memory = False
use_amp = False
train_ratio = 0.9

device = torch.device("cuda")

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers
    from config import GPT_SMALL_CONFIG, the_verdict_path
    from llm_quest.dataset import create_dataloader
    from llm_quest.engine import training_eval_loop
    from llm_quest.gpt.gpt_model import GPTModel

    # --- Data preparation ---
    with open(the_verdict_path, "r") as file:
        txt = file.read()

    # train, val split
    split = int(train_ratio * len(txt))
    train_data = txt[:split]
    val_data = txt[split:]

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Loaders ---
    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=GPT_SMALL_CONFIG["context_length"],
        stride=GPT_SMALL_CONFIG["context_length"],
        tokenizer=tokenizer,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=GPT_SMALL_CONFIG["context_length"],
        stride=GPT_SMALL_CONFIG["context_length"],
        tokenizer=tokenizer,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- Training ---
    model = GPTModel(GPT_SMALL_CONFIG)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)

    training_eval_loop(
        train_loader,
        val_loader,
        model=model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        warmup_percent=warmup_percent,
        init_lr=init_lr,
        peak_lr=peak_lr,
        min_lr=min_lr,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        device=device,
        use_amp=use_amp,
    )
    ## saving final model and optimizer parameters
    # torch.save(
    #    {
    #        "model_state_dict": model.state_dict(),
    #        #"optimizer_state_dict": optimizer.state_dict(),
    #    },
    #    custom_pretrained_w_gpt2,
    # )
