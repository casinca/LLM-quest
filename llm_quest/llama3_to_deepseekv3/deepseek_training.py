from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config
from llm_quest.dataset import HFDataset
from llm_quest.engine import LearningRateScheduler
from llm_quest.llama3_to_deepseekv3.custom_collate_mtp import collate_function_mtp
from llm_quest.llama3_to_deepseekv3.deepseek_engine import training_eval_loop_mtp
from llm_quest.llama3_to_deepseekv3.deepseek_model import DeepSeekV3Model

torch.manual_seed(123)

# quick hparams for testing
num_epoch = 2
peak_lr = 5e-4
init_lr = 1e-5
min_lr = 1e-5
decay = "cosine"
eval_freq = 5
eval_iter = 5
weight_decay = 0.1
batch_size = 16
device = config.auto_device
deepseek_small_cfg = config.DEEPSEEK_SMALL_CONFIG

tokenizer = tiktoken.get_encoding("gpt2")

train_hf = HFDataset(config.fineweb_train, tokenizer=tokenizer, max_samples=3_200)
val_hf = HFDataset(config.fineweb_val, tokenizer=tokenizer, max_samples=3_200 * 0.1)

custom_collate = partial(collate_function_mtp, custom_max_len=deepseek_small_cfg["context_length"], device="cpu")

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
model = DeepSeekV3Model(deepseek_small_cfg)
model.bfloat16().to(device)

# no need to set optimizer's lr, the LR scheduler will init optimizer's lr
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, fused=True)

total_steps = len(train_loader) * num_epoch
warmup_steps = int(0.2 * total_steps)  # matching old warmup_percent arg of 20%
lr_scheduler = LearningRateScheduler(
    optimizer,
    total_steps=total_steps,
    init_lr=init_lr,
    peak_lr=peak_lr,
    warmup_steps=warmup_steps,
    min_lr=min_lr,
    decay=decay,
)

train_loss, val_loss = training_eval_loop_mtp(
    train_loader,
    val_loader,
    model=model,
    optimizer=optimizer,
    num_epoch=num_epoch,
    lr_scheduler=lr_scheduler,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
    device=device,
    use_amp=False,
)
