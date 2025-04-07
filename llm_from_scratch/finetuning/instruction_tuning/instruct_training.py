from functools import partial

import tiktoken
import torch
from torch.utils.data import DataLoader

import config
from gpt_download import download_and_load_gpt2
from llm_from_scratch.GPT.gpt_model import GPTModel
from llm_from_scratch.dataset import InstructionDataset, collate_function
from llm_from_scratch.engine import training_eval_loop
from llm_from_scratch.utils import load_weights_into_gpt

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
batch_size = 8

data_device = "cpu"
model_device = "cuda"

model_cfg = config.config_creator("gpt_m")  # using Medium sized GPT config
settings, params = download_and_load_gpt2(model_size="355M", models_dir=config.openai_pretrained_w_gpt2_m)
model = GPTModel(model_cfg)
load_weights_into_gpt(model, params)

train_set = InstructionDataset(config.instruct_train_path, tokenizer)
val_set = InstructionDataset(config.instruct_val_path, tokenizer)

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
    num_workers=0,
    pin_memory=False,
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    collate_fn=custom_collate,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=False,
)


if __name__ == "__main__":
    num_epoch = 2
    peak_lr = 5e-4

    model.to(model_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    training_eval_loop(
        train_loader,
        val_loader,
        model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        warmup_percent=0.0,
        init_lr=0,
        peak_lr=peak_lr,
        min_lr=5e-4,
        eval_freq=5,
        eval_iter=5,
        device=model_device,
        use_amp=False,
    )

    # save instruct finetuned model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        config.ft_instruct_w_gpt2,
    )
