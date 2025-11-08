from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from llm_quest.dataset import HFDataset, collate_function
from llm_quest.engine import training_eval_loop
from llm_quest.qwen.qwen3_next.qwen3_next_model import Qwen3NextModel

# hyperparameters
torch.manual_seed(123)
num_epoch = 2
peak_lr = 5e-4
init_lr = 1e-5
min_lr = 1e-5
eval_freq = 5
eval_iter = 5
warmup_percent = 0.2
weight_decay = 0.1
batch_size = 4
use_amp = False

qwen3_next_cfg = config.QWEN3_NEXT_SMALL_CONFIG
qwen3_next_cfg["training"] = True

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

# Dataset setup (using FineWeb samples)
train_hf = HFDataset(config.fineweb_train, tokenizer=tokenizer, max_samples=3_200)
val_hf = HFDataset(config.fineweb_val, tokenizer=tokenizer, max_samples=3_200 * 0.1)

custom_collate = partial(
    collate_function,
    custom_max_len=qwen3_next_cfg["context_length"],
    device="cpu",
)

train_loader = DataLoader(
    train_hf,
    batch_size=batch_size,
    collate_fn=custom_collate,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_hf,
    batch_size=batch_size,
    collate_fn=custom_collate,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

# Initialize model and training setup
device = torch.device("cuda")
model = Qwen3NextModel(qwen3_next_cfg)
model.bfloat16().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay, fused=True)

print("Starting Qwen3 Next training...")
train_loss, val_loss = training_eval_loop(
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
