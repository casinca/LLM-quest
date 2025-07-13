import json

import tiktoken
import torch

import config
from llm_quest.gpt.generate import generate_loop
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import alpaca_deepseek_format, ids_to_text, text_to_ids

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
data_device = "cpu"
model_device = "cuda"

test_txt = []
with open("../../../data/processed_data/gsm8k_processed/gsm8k_test.jsonl", "r") as f:
    for line in f:
        test_txt.append(json.loads(line))

model_cfg = config.config_creator("gpt_m")
model = GPTModel(model_cfg)

# loading our SFT model params
ft_checkpoint = torch.load(config.sft_reasoning_gpt2, weights_only=True)
model.load_state_dict(ft_checkpoint["model_state_dict"])
model.to(model_device)
model.eval()

for instruct in test_txt[0:4]:
    input_txt = alpaca_deepseek_format(instruct, include_response=False)

    output = generate_loop(
        text_to_ids(input_txt, tokenizer),
        model,
        max_gen=250,
        context_length=model_cfg["context_length"],
        temp=1,
        top_k=10,
        eos_id=50256,
        device=model_device,
    )

    generated_text = ids_to_text(output, tokenizer)
    response_text = generated_text[len(input_txt) :].replace("### Response:", "").strip()
    print(input_txt)
    reasoning_part, separator, answer_part = instruct["answer"].partition("\n#### ")
    answer_formatted = f"<think>{reasoning_part}</think> <answer>{answer_part}</answer>"
    print(f"\nCorrect response:\n{answer_formatted}")
    print(f"\nModel response:\n{response_text.strip()}")
    print("-------------------------------------")
