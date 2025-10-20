import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTModel

import config
from llm_quest.dataset import MultimodalDataset
from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.multimodal.multimodal_engine import multimodal_training_loop_simple
from llm_quest.vit.vit_engine import ViTAdapter

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gpt_size = "gpt_s"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds_train = load_dataset("jxie/flickr8k", split="train")
train_dataset = MultimodalDataset(ds_train, tokenizer, image_size=224, max_caption_len=60)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

ds_val = load_dataset("jxie/flickr8k", split="validation")
val_dataset = MultimodalDataset(ds_val, tokenizer, image_size=224, max_caption_len=60)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize models and adapter
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")

gpt_config = config.config_creator(gpt_size)
multimodal_model = GPTModel(gpt_config)
weights_path = download_gpt_model(gpt_size=gpt_size, save_dir=config.openai_pretrained_w_gpt2_s)
load_gpt_weights(multimodal_model, weights_path)

adapter = ViTAdapter(
    vit_d_out=vit_model.config.hidden_size,  # 768
    llm_d_in=gpt_config["emb_dim"],
    adapter_type="ffn",
)

# Optimizer
trainable_params = list(multimodal_model.parameters()) + list(adapter.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

print("Starting training...")
mm_model, mm_adapter = multimodal_training_loop_simple(
    vit_model=vit_model,
    multimodal_model=multimodal_model,
    adapter=adapter,
    train_loader=train_loader,
    optimizer=optimizer,
    num_epochs=1,
    device=device,
    hf_vit_model=True,
    val_loader=val_loader,
    eval_freq=100,
    eval_iter=5,
)

torch.save(mm_model.state_dict(), config.multimodal_gpt)
torch.save(mm_adapter.state_dict(), config.multimodal_adapter)
