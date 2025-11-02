from functools import partial

import torch
from torch.utils.data import DataLoader

torch.manual_seed(123)

# --- Hyperparameters ---
batch_size = 8
num_epoch = 2
peak_lr = 5e-4
warmup_percent = 0.0
init_lr = 0
min_lr = 5e-4
eval_freq = 5
eval_iter = 5
weight_decay = 0.1
num_workers = 0
pin_memory = False
use_amp = False

# LoRA hyperparameters
rank = 4
alpha = 16

data_device = "cpu"
model_device = "cuda"

if __name__ == "__main__":
    # heavy imports inside if __name__ == "__main__" for num_workers
    import tiktoken

    import config
    from llm_quest.common.lora import LoRALinearLayer
    from llm_quest.dataset import InstructionDataset, collate_function
    from llm_quest.engine import training_eval_loop
    from llm_quest.gpt.gpt_attention import MultiHeadAttention
    from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
    from llm_quest.gpt.gpt_model import GPTModel

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- Loaders ---
    model_cfg = config.gpt2_config_creator("gpt_m")  # using Medium sized gpt config

    weights_path = download_gpt_model(gpt_size="gpt_m", save_dir=config.openai_pretrained_w_gpt2_m)
    model = GPTModel(model_cfg)
    load_gpt_weights(model, weights_path)

    ############ LoRA changes from instruct_training.py ############

    # freezing weights (only LoRA weights will get the updates, see note end of block)
    for param in model.parameters():
        param.requires_grad = False

    # replacing only Linear layers within Attention modules (attention weights (Q, K, V) + output layer)
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            for attr in ["w_queries", "w_keys", "w_values", "out_proj"]:
                linear = getattr(module, attr)
                setattr(
                    module,
                    attr,
                    LoRALinearLayer(
                        linear.in_features, linear.out_features, rank, alpha, linear_bias=linear.bias is not None
                    ),
                )

    # Note: Concerning frozen and trainable params when replacing nn.Linear. They were already part of the
    # model and will be picked up and frozen automatically, whereas A and B are nn.Parameters (default grad=True) and not
    # part of the original model, thus will not be picked up by the freeze and maintain their default grad=True which is
    # what we want.

    ########################

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)

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
        use_amp=use_amp,
    )

    ## save instruct finetuned model
    # torch.save(
    #    {
    #        "model_state_dict": model.state_dict(),
    #        #"optimizer_state_dict": optimizer.state_dict(),
    #    },
    #    config.ft_instruct_w_gpt2,
    # )
