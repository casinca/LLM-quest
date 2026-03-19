"""
This is a text only generation, to make sure the text part of the VLM worked as expected before doing
everything all at once.

Updated for Cache support using MRoPE, or could be used without Cache with standard 1D RoPE
"""

import torch
from transformers import AutoTokenizer

from config import QWEN3_5_08B_CONFIG, auto_device
from llm_quest.generate import sampling
from llm_quest.qwen.qwen3_5.qwen3_5_text_model import Qwen3_5TextModel
from llm_quest.qwen.qwen3_5.qwen3_5_weight_loading import load_qwen3_5_text_weights
from llm_quest.utils import Qwen3_5Cache

###########
# Hparams #
###########

enable_thinking = True
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

print(f"\nPrompt: '{prompt}'\n")

message = [{"role": "user", "content": prompt}]

# not sure
formatted_prompt = tokenizer.apply_chat_template(
    message, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking
)

input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
eos_token_id = tokenizer.eos_token_id

# Init Cache
cache = Qwen3_5Cache(
    n_layers=model_cfg["n_layers"],
    linear_sdpa_ratio=model_cfg["linear_sdpa_ratio"],
    prompt_len=input_ids.shape[-1],
    context_len=model_cfg["context_length"],
)

# For MRoPE text-only: position_ids is (3, batch, seq_len) where all 3 dimensions are the same
seq_len = input_ids.shape[-1]
position_ids_1d = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
position_ids = position_ids_1d.unsqueeze(0).expand(3, -1, -1)  # (3, 1, seq_len)

generated_token_ids = []

############################
# Single batch generation #
############################

# Doing manual generation loop as generate_loop_kv_cache() would need extra refactoring to support Qwen3_5Cache
torch.manual_seed(seed)
with torch.inference_mode():
    # Prefill/ builds the cache
    logits = qwen3_5_model(input_ids, position_ids=position_ids, cache=cache)[:, -1, :]

    # generate tokens one at a time with cache
    next_pos = seq_len  # tracking position for RoPE
    for _ in range(max_gen):
        next_token = sampling(logits, topk, topp, min_p, temp)

        if next_token.item() == eos_token_id:
            break

        generated_token_ids.append(next_token)

        # Position IDs for next token, shape (3, 1, 1) all the same `next_pos` value ie T=H=W=next_pos
        next_position_ids = torch.tensor([[[next_pos]]], device=device).expand(3, -1, -1)

        logits = qwen3_5_model(next_token, position_ids=next_position_ids, cache=cache).squeeze(1)
        next_pos += 1


output = torch.cat([input_ids] + generated_token_ids, dim=-1)
generated_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
print(f"Generated:\n{generated_text}")
