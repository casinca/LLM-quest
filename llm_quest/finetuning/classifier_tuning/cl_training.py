import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from llm_quest.dataset import SpamDataset
from llm_quest.finetuning.classifier_tuning.cl_engine import classifier_training_eval_loop
from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
from llm_quest.gpt.gpt_model import GPTModel

# init params
tokenizer = tiktoken.get_encoding("gpt2")
batch_size = 8
torch.manual_seed(123)

# importing dataset
train_path = config.spam_train_path
val_path = config.spam_val_path
train_set = SpamDataset(file=train_path, tokenizer=tokenizer)
val_set = SpamDataset(file=val_path, max_length=train_set.max_length, tokenizer=tokenizer)


# create loaders
# could have used our wrapper create_dataloader() function alternatively
train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=False, num_workers=0)


# pretrained model params
weights_path = download_gpt_model(gpt_size="gpt_s", save_dir=config.openai_pretrained_w_gpt2_s)

# model config
model_config = config.config_creator("gpt_s")
model_config["drop_rate"] = 0.0

model = GPTModel(model_config)
load_gpt_weights(model, weights_path)

# freeze model - make all layers non-trainable
for param in model.parameters():
    param.requires_grad = False

# prepare model for binary classification (or multi class)
num_classes = 2
# shrinking the "out_features" projection of the "out" linear layer (head) to "num_classes"
model.out = nn.Linear(model_config["emb_dim"], num_classes)
# unfreezing only the last trf block + LayerNorm
# note for completeness: by default recreating a nn.Linear layer has grad=True no need to re-specify for the out_layer
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_ln.parameters():
    param.requires_grad = True


if __name__ == "__main__":

    num_epoch = 5
    peak_lr = 5e-4

    device = "cuda"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)

    classifier_training_eval_loop(
        train_loader,
        val_loader,
        model=model,
        optimizer=optimizer,
        num_epoch=num_epoch,
        warmup_percent=0.0,
        init_lr=0,
        peak_lr=peak_lr,
        min_lr=peak_lr,
        eval_freq=5,
        eval_iter=5,
        device=device,
    )

    # saving final model and optimizer parameters
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": optimizer.state_dict(),
        },
        config.ft_classifier_w_gpt2,
    )
