import tiktoken
import torch

from config import GPT_SMALL_CONFIG, the_verdict_path
from llm_quest.dataset import create_dataloader
from llm_quest.engine import training_eval_loop
from llm_quest.gpt.gpt_model import GPTModel

with open(the_verdict_path, "r") as file:
    txt = file.read()

# train, val split
train_ratio = 0.9
split = int(train_ratio * len(txt))
train_data = txt[:split]
val_data = txt[split:]

tokenizer = tiktoken.get_encoding("gpt2")

# creating loaders
train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_SMALL_CONFIG["context_length"],
    stride=GPT_SMALL_CONFIG["context_length"],
    tokenizer=tokenizer,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_SMALL_CONFIG["context_length"],
    stride=GPT_SMALL_CONFIG["context_length"],
    tokenizer=tokenizer,
    shuffle=False,
    drop_last=True,
    num_workers=0,
)

if __name__ == "__main__":
    torch.manual_seed(123)

    num_epoch = 10
    peak_lr = 5e-4
    # setting device and putting model on it
    device = torch.device("cuda")
    model = GPTModel(GPT_SMALL_CONFIG)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    training_eval_loop(
        train_loader,
        val_loader,
        model=model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        warmup_percent=0.2,
        init_lr=1e-5,
        peak_lr=peak_lr,
        min_lr=1e-5,
        eval_freq=5,
        eval_iter=5,
        device=device,
        use_amp=False,
    )
    ## saving final model and optimizer parameters
    # torch.save(
    #    {
    #        "model_state_dict": model.state_dict(),
    #        "optimizer_state_dict": optimizer.state_dict(),
    #    },
    #    custom_pretrained_w_gpt2,
    # )
