import json

import tiktoken
import torch

import config
from llm_quest.gpt.generate import generate_loop
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import alpaca_prompt_format, ids_to_text, text_to_ids

# the training result showed a discrepancy between the training loss and the val loss. The delta increases after
# a certain amount of steps/time that clearly hints at overfitting.
# This is not unexpected for multiple reasons, especially these 2 coupled:
# - We did a full finetuning (unlike for the classifier finetuning with mostly frozen layers/PEFT)
# - The toyset is relatively small

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
data_device = "cpu"
model_device = "cuda"

with open(config.instruct_alpaca_test_path, "r") as f:
    test_txt = json.load(f)

model_cfg = config.config_creator("gpt_m")
model = GPTModel(model_cfg)
# loading our finetuned model params
ft_checkpoint = torch.load(config.ft_instruct_w_gpt2, weights_only=True)
model.load_state_dict(ft_checkpoint["model_state_dict"])
model.to(model_device)
model.eval()

for instruct in test_txt[0:2]:

    input_txt = alpaca_prompt_format(instruct, include_output=False)

    output = generate_loop(
        text_to_ids(input_txt, tokenizer),
        model,
        max_gen=250,
        context_length=model_cfg["context_length"],
        temp=1,
        top_k=20,
        eos_id=50256,
        device=model_device,
    )

    generated_text = ids_to_text(output, tokenizer)
    response_text = generated_text[len(input_txt) :].replace("### Response:", "").strip()
    print(input_txt)
    print(f"\nCorrect response:\n>> {instruct['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")
