import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTModel

import config
from llm_quest.dataset import MultimodalDataset
from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.multimodal.multimodal_engine import vlm_training_loop_simple
from llm_quest.vit.vit_engine import ViTAdapter

# Hyperparameters
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gpt_size = "gpt_s"
num_epochs = 2
batch_size = 8
max_caption_len = 60
vlm_drop_rate = 0.2
adapter_drop_rate = 0.0
learning_rate = 1e-5
eval_freq = 100
eval_iter = 5

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds_train = load_dataset("jxie/flickr8k", split="train")
train_dataset = MultimodalDataset(ds_train, tokenizer, image_size=224, max_caption_len=max_caption_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

ds_val = load_dataset("jxie/flickr8k", split="validation")
val_dataset = MultimodalDataset(ds_val, tokenizer, image_size=224, max_caption_len=max_caption_len)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models and adapter
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

gpt_config = config.config_creator(gpt_size)
gpt_config["drop_rate"] = vlm_drop_rate
vlm_model = GPTModel(gpt_config)
weights_path = download_gpt_model(gpt_size=gpt_size, save_dir=config.openai_pretrained_w_gpt2_s)
load_gpt_weights(vlm_model, weights_path)

adapter = ViTAdapter(
    vit_d_out=vit_model.config.hidden_size,  # 768
    llm_d_in=gpt_config["emb_dim"],
    adapter_type="ffn",
    dropout=adapter_drop_rate,
)

# Optimizer
trainable_params = list(vlm_model.parameters()) + list(adapter.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

print("Starting training...")
vlm_model, vit_adapter = vlm_training_loop_simple(
    vit_model=vit_model,
    vlm_model=vlm_model,
    adapter=adapter,
    train_loader=train_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    hf_vit_model=True,
    val_loader=val_loader,
    eval_freq=eval_freq,
    eval_iter=eval_iter,
)

torch.save(vlm_model.state_dict(), config.vlm_gpt)
torch.save(vit_adapter.state_dict(), config.vlm_adapter)
