import tiktoken
import torch

import config
from gpt_download import download_and_load_gpt2
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.speculative_decoding.spec_decoding_engine import speculative_generate
from llm_quest.utils import load_weights_into_gpt, text_to_ids

# --- Hyperparameters ---
max_gen = 40
draft_max_gen = 10
target_context_length = 1024  # TODO switch to config
draft_context_length = 768

prompt = "This is where it"

if __name__ == "__main__":

    target_settings, target_params = download_and_load_gpt2(
        model_size="355M", models_dir=config.openai_pretrained_w_gpt2_m
    )
    draft_settings, draft_params = download_and_load_gpt2(
        model_size="124M", models_dir=config.openai_pretrained_w_gpt2_s
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    target_model_config = config.config_creator("gpt_m")
    draft_model_config = config.config_creator("gpt_s")

    torch.manual_seed(123)

    device = "cuda"

    target_model = GPTModel(target_model_config)
    draft_model = GPTModel(draft_model_config)

    load_weights_into_gpt(target_model, target_params)
    load_weights_into_gpt(draft_model, draft_params)

    target_model.to(device=device, dtype=torch.bfloat16).eval()
    draft_model.to(device=device, dtype=torch.bfloat16).eval()

    speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt=text_to_ids(prompt, tokenizer=tokenizer),
        max_gen=max_gen,
        context_length=draft_context_length,  # TODO need both?
        draft_max_gen=draft_max_gen,
        top_k=20,
        top_p=1.0,
        temp=1.0,
        eos_id=50256,
        device=device,
    )
