import torch
from transformers import AutoTokenizer

from config import qwen3_config_creator
from llm_quest.generate import (
    generate_batched_loop_kv_cache,
    generate_batched_loop_kv_cache_left_pad,
    generate_loop,
    generate_loop_kv_cache,
)
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model
from llm_quest.qwen.qwen3.qwen3_weight_loading import load_qwen3_weights

# Sometimes No KVcache output differs from KVcache output.
# This is expected, same thing happens when comparing to other baselines (HF + @rasbt)
# Detailed explanations on causes, from a top HF engineer:
# https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535


###########
# Hparams #
###########
device = "cuda"

base_model = True
enable_thinking = False
add_generation_prompt = False  # more suited for base_model=False: adds the assistant token at the end of the prompt

max_gen = 80
topk = 25
topp = 0.95
temp = 0.0
seed = 123
pad_side = "right"
qwen3_cfg = qwen3_config_creator("0.6B", base_model=base_model)

prompt = "The capital of France is"
batch_prompts = [
    "The quick brown fox jumps over the lazy dog and then goes to the park to play with",
    "The capital of France is",
]

########
# Prep #
########
batched_generation_func = (
    generate_batched_loop_kv_cache_left_pad if pad_side == "left" else generate_batched_loop_kv_cache
)

tokenizer = AutoTokenizer.from_pretrained(qwen3_cfg["model_path"])
qwen3_model = Qwen3Model(qwen3_cfg)
qwen3_model = load_qwen3_weights(qwen3_model, qwen3_cfg)
qwen3_model.to(device).eval()

# Apply the ChatML template to format the conversation properly if needed
if not base_model:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
else:
    formatted_prompt = prompt

input_tensor = torch.tensor([tokenizer.encode(formatted_prompt)]).to(device)


############################
# Single batch generation #
###########################
print("\n\n ######## Testing single generation ######## \n\n")

torch.manual_seed(seed)
output = generate_loop_kv_cache(
    input_tensor=input_tensor,
    model=qwen3_model,
    max_gen=max_gen,
    context_length=qwen3_cfg["context_length"],
    top_k=topk,
    top_p=topp,
    temp=temp,
    eos_id=tokenizer.eos_token_id,
    rope_model=True,
)

print(tokenizer.decode(output[0].tolist(), skip_special_tokens=False))


########################################################
# Batch generation with left padding or right padding #
########################################################
print("\n\n ######## Testing batch generation with left padding or right padding ######## \n\n")

tokenizer = AutoTokenizer.from_pretrained(qwen3_cfg["model_path"], padding_side=pad_side)
batch_encoded = tokenizer.batch_encode_plus(
    batch_prompts,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=qwen3_cfg["context_length"],
    truncation=True,
    padding=True,
)

last_real = torch.sum(batch_encoded.attention_mask, dim=1) - 1

print("Batch encoded input IDs:")
print(batch_encoded.input_ids)
print("Batch encoded attention mask:")
print(batch_encoded.attention_mask)
print("last_real:", last_real)

batched_input_ids = batch_encoded.input_ids.to(device)
batched_attention_mask = batch_encoded.attention_mask.to(device)

torch.manual_seed(seed)
batched_output = batched_generation_func(
    input_tensor=batched_input_ids,
    model=qwen3_model,
    max_gen=max_gen,
    context_length=qwen3_cfg["context_length"],
    top_k=topk,
    top_p=topp,
    temp=temp,
    eos_id=tokenizer.eos_token_id,
    device=device,
    attention_mask=batched_attention_mask,
)

for i, output in enumerate(batched_output):
    print(f"\nPrompt {i+1}:\n")
    print(tokenizer.decode(output.tolist(), skip_special_tokens=False))
