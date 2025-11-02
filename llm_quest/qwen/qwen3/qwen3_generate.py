import torch
from transformers import AutoTokenizer

from config import qwen3_config_creator
from llm_quest.gpt.generate import generate_loop, generate_loop_kv_cache
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model
from llm_quest.qwen.qwen3.qwen3_weight_loading import load_qwen3_weights

# Sometimes No KVcache output differs from KVcache output.
# This is expected, same thing happens when comparing to other baselines (HF + @rasbt)
# Detailed explanations on causes, from a top HF engineer:
# https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

torch.manual_seed(123)
device = "cuda"
qwen3_cfg = qwen3_config_creator("0.6B", base_model=True)

tokenizer = AutoTokenizer.from_pretrained(qwen3_cfg["model_path"])
qwen3_model = Qwen3Model(qwen3_cfg)
qwen3_model = load_qwen3_weights(qwen3_model, qwen3_cfg)

qwen3_model.to(device).eval()

prompt = "Give me a short introduction to large language models."
input_tensor = torch.tensor([tokenizer.encode(prompt)]).to(device)

output = generate_loop_kv_cache(
    input_tensor=input_tensor,
    model=qwen3_model,
    max_gen=50,
    context_length=qwen3_cfg["context_length"],
    top_k=25,
    top_p=0.95,
    temp=1.4,
    eos_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0].tolist(), skip_special_tokens=False))
