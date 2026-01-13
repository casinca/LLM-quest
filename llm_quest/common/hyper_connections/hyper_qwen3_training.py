import math
from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from llm_quest.common.hyper_connections.hyper_qwen3 import HyperQwen3Model
from llm_quest.dataset import HFDataset, collate_function
from llm_quest.engine import LearningRateScheduler, training_eval_loop

torch.manual_seed(123)
nano_qwen_config = {
    "vocab_size": 151936,  # 151936 # 50304
    "rope_base": 1_000_000,
    "dtype": torch.bfloat16,
    "model_type": "dense",
    "emb_dim": 512,
    "head_dim": 64,
    "n_layers": 12,
    "n_heads": 8,
    "num_kv_groups": 4,
    "hidden_dim": 2048,
    "context_length": 512,
    "tie_embeddings": True,
}

hparams = {
    "expansion_rate": 4,
    "num_epoch": 1,
    "peak_lr": 5e-4,
    "init_lr": 1e-5,
    "min_lr": 1e-5,
    "decay": "cosine",
    "eval_freq": 10,
    "eval_iter": 2,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "batch_size": 8,
    "accumulation_steps": 1,
    "use_amp": False,
}


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# tokenizer = tiktoken.get_encoding("gpt2")

train_hf = HFDataset(config.fineweb_train, tokenizer=tokenizer, max_samples=10_000)
val_hf = HFDataset(config.fineweb_val, tokenizer=tokenizer, max_samples=200)

custom_collate = partial(
    collate_function,
    custom_max_len=nano_qwen_config["context_length"],
    device="cpu",
)

train_loader = DataLoader(
    train_hf,
    batch_size=hparams["batch_size"],
    collate_fn=custom_collate,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_hf,
    batch_size=hparams["batch_size"],
    collate_fn=custom_collate,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)

device = config.auto_device
model = HyperQwen3Model(nano_qwen_config, expansion_rate=hparams["expansion_rate"])
# For simplicity casting the whole model to bf16, technically hyperconnections should be in fp32
model.bfloat16().to(device)

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=hparams["weight_decay"], fused=True)

update_steps = math.ceil(len(train_loader) / hparams["accumulation_steps"])
total_steps = update_steps * hparams["num_epoch"]
warmup_steps = int(0.2 * total_steps)

lr_scheduler = LearningRateScheduler(
    optimizer,
    total_steps=total_steps,
    init_lr=hparams["init_lr"],
    peak_lr=hparams["peak_lr"],
    warmup_steps=warmup_steps,
    min_lr=hparams["min_lr"],
    decay=hparams["decay"],
)

training_eval_loop(
    train_loader,
    val_loader,
    model=model,
    optimizer=optimizer,
    num_epoch=hparams["num_epoch"],
    lr_scheduler=lr_scheduler,
    eval_freq=hparams["eval_freq"],
    eval_iter=hparams["eval_iter"],
    device=device,
    accumulation_steps=hparams["accumulation_steps"],
    use_amp=hparams["use_amp"],
)
