from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config
from config import DEEPSEEK_SMALL
from llm_quest.dataset import HFDataset
from llm_quest.llama3_to_deepseekv3.custom_collate_mtp import collate_function_mtp
from llm_quest.llama3_to_deepseekv3.deepseek_engine import training_eval_loop_mtp
from llm_quest.llama3_to_deepseekv3.deepseek_model import DeepSeekV3Model

# quick hparams for testing
num_epoch = 2
peak_lr = 5e-4
init_lr = 1e-5
min_lr = 1e-5
eval_freq = 5
eval_iter = 5
warmup_percent = 0.2
weight_decay = 0.1
batch_size = 16

tokenizer = tiktoken.get_encoding("gpt2")

train_hf = HFDataset(config.fineweb_train, tokenizer=tokenizer, max_samples=3_200)
val_hf = HFDataset(config.fineweb_val, tokenizer=tokenizer, max_samples=3_200 * 0.1)

custom_collate = partial(collate_function_mtp, custom_max_len=DEEPSEEK_SMALL["context_length"], device="cpu")

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

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
model = DeepSeekV3Model(DEEPSEEK_SMALL)
model.bfloat16().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay, fused=True)

train_loss, val_loss = training_eval_loop_mtp(
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
    use_amp=False,
)
