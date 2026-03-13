import torch
from datasets import load_dataset
from torchvision.transforms import functional as F
from transformers import AutoTokenizer

from config import QWEN3_5_08B_CONFIG, auto_device
from llm_quest.generate import generate_loop
from llm_quest.qwen.qwen3_5.qwen3_5_vlm_model import Qwen3_5VLM
from llm_quest.qwen.qwen3_5.qwen3_5_weight_loading import load_qwen3_5_vlm_weights

###################################################
# Hparams

max_gen = 50
temp = 1.0
seed = 123

model_cfg = QWEN3_5_08B_CONFIG

image_size = model_cfg["img_width"]  # square H=W reduced to 384x384 but can go up to 768x768
tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])
patch_size = model_cfg["patch_size"]
temporal_patch_size = model_cfg["temporal_patch_size"]

device = auto_device
print(f"\nUsing DEVICE: {device.type}\n")

###################################################
# Load the first image of the test set and visualize it (same image used for Multimodal GPT-2 test)

print("Loading test image from flickr8k...")
ds_test = load_dataset("jxie/flickr8k", split="test")
image = ds_test[0]["image"]
print(f"Image size: {image.size}")
image.show()  # 2 dogs fighting/playing in the snow

###################################################
# Preprocess image manually (not using HF processor but could) for `PatchEmbedding3D` format: 5D tensor (B, C, T, H, W)

img = F.resize(image, (image_size, image_size))
tensor = F.to_tensor(img)  # (3, 384, 384), range [0, 255] → [0, 1]
tensor = F.normalize(tensor, mean=model_cfg["image_mean"], std=model_cfg["image_std"])  # [-1, 1]

# duplicate for temporal dimension: (temporal_patch_size, C, H, W)
video_tensor = tensor.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)  # (2, 3, 384, 384)
video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, 2, 384, 384)

print(f"Image pixels shape (5D): {video_tensor.shape}")

###################################################
# Precompute the complete input token sequence (with image placeholders token IDs)

prompt_text = "What do you see in the image?"

n_patches_h = image_size // patch_size
n_patches_w = image_size // patch_size
n_merged_patches = (n_patches_h * n_patches_w) // (model_cfg["spatial_merge_size"] ** 2)
print(f"Number of image placeholder tokens: {n_merged_patches}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "dummy_image"},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# augment prompt with chat template and image placeholder tokens
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_pad_token_str = tokenizer.convert_ids_to_tokens(model_cfg["image_token_id"])
text = text.replace(image_pad_token_str, image_pad_token_str * n_merged_patches)

text_ids = tokenizer.encode(text, add_special_tokens=False)
input_ids = torch.tensor([text_ids])
print(f"Input shape: {input_ids.shape}  (total tokens: {len(text_ids)})")

###################################################
# Load model + weights

print("\nLoading Qwen3_5VLM model")
vlm_model = Qwen3_5VLM(model_cfg)

total_params = sum(p.numel() for p in vlm_model.parameters())
vision_params = sum(p.numel() for p in vlm_model.vision_model.parameters())
text_params = sum(p.numel() for p in vlm_model.language_model.parameters())
print(f"Total params: {total_params:,}")
print(f"Vision: {vision_params:,}")
print(f"Text: {text_params:,}")

vlm_model = load_qwen3_5_vlm_weights(vlm_model, model_cfg)
vlm_model.to(device).eval()

###################################################
# Simple Generation

print(f"\nGenerating (max {max_gen} tokens)")
print(f"Prompt: '{prompt_text}'\n")

input_ids = input_ids.to(device)
image_pixels_5d = video_tensor.to(device)

torch.manual_seed(seed)


def vlm_arg_wrapper(input_tensor):
    """makes generate_loop() compatible with the VLM model, for image pixels arg"""
    return vlm_model(input_tensor, image_pixels=image_pixels_5d)


generated_ids = generate_loop(
    input_tensor=input_ids,
    model=vlm_arg_wrapper,
    max_gen=max_gen,
    context_length=model_cfg["context_length"],
    top_k=None,
    top_p=None,
    min_p=None,
    temp=temp,
    eos_ids=tokenizer.eos_token_id,
    device=device,
)

generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
print(f"Generated:\n{generated_text}")
