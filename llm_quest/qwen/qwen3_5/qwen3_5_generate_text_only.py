"""
This is a text only generation, to make sure the text part of the VLM worked as expected before doing
everything all at once.
"""

import torch
from transformers import AutoTokenizer

from config import QWEN3_5_08B_CONFIG, auto_device
from llm_quest.generate import generate_loop
from llm_quest.qwen.qwen3_5.qwen3_5_text_model import Qwen3_5TextModel
from llm_quest.qwen.qwen3_5.qwen3_5_weight_loading import load_qwen3_5_text_weights

###########
# Hparams #
###########

base_model = False
enable_thinking = False
add_generation_prompt = False  # more suited for base_model=False: adds the assistant token at the end of the prompt

max_gen = 120
topk = 20
topp = 0.95
min_p = None
temp = 1.0
seed = 123

device = auto_device
print(f"\nUsing DEVICE: {device.type}\n")

model_cfg = QWEN3_5_08B_CONFIG
tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"])

prompt = "Give me a short introduction to large language models."

########
# Prep #
########

qwen3_5_model = Qwen3_5TextModel(model_cfg)
qwen3_5_model = load_qwen3_5_text_weights(qwen3_5_model, model_cfg)
qwen3_5_model.to(device).eval()

############################
# Single batch generation #
############################
print(f"\nPrompt: '{prompt}'\n")

message = [{"role": "user", "content": prompt}]

formatted_prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
)

input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
eos_token_id = tokenizer.eos_token_id

torch.manual_seed(seed)
output = generate_loop(
    input_tensor=input_ids,
    model=qwen3_5_model,
    max_gen=max_gen,
    context_length=model_cfg["context_length"],
    top_k=topk,
    top_p=topp,
    min_p=min_p,
    temp=temp,
    eos_ids=eos_token_id,
    device=device,
)

generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
print(f"Generated:\n{generated_text}")
