import tiktoken
import torch

import config
from gpt_download import download_and_load_gpt2
from llm_quest.gpt.generate import generate_loop_kv_cache
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.speculative_decoding.spec_decoding_engine import speculative_generate
from llm_quest.utils import ids_to_text, load_weights_into_gpt, text_to_ids

# --- Hyperparameters ---
max_gen = 20
draft_max_gen = 3
target_context_length = 1024  # TODO switch to config
draft_context_length = 768
seed = 123
temp = 1.4
top_k = 25
top_p = None

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

    device = "cuda"

    target_model = GPTModel(target_model_config)
    draft_model = GPTModel(draft_model_config)

    load_weights_into_gpt(target_model, target_params)
    load_weights_into_gpt(draft_model, draft_params)

    target_model.to(device=device, dtype=torch.bfloat16).eval()
    draft_model.to(device=device, dtype=torch.bfloat16).eval()
    torch.manual_seed(seed)
    output = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt=text_to_ids(prompt, tokenizer=tokenizer),
        max_gen=max_gen,
        context_length=draft_context_length,  # TODO need both?
        draft_max_gen=draft_max_gen,
        top_k=top_k,
        top_p=top_p,
        temp=temp,
        eos_id=50256,
        device=device,
    )
    print("\nOutput from speculative decoding:")
    print(output)
    print("\n")
    print(ids_to_text(output, tokenizer))

    print("-------------")

    #    torch.manual_seed(seed)
    #    output2 = speculative_generate(
    #        target_model=target_model,
    #        draft_model=target_model,
    #        prompt=text_to_ids(prompt, tokenizer=tokenizer),
    #        max_gen=max_gen,
    #        context_length=draft_context_length,
    #        draft_max_gen=draft_max_gen,
    #        top_k=top_k,
    #        top_p=top_p,
    #        temp=temp,
    #        eos_id=50256,
    #        device=device,
    #    )
    #    print("\nOutput from speculative decoding:")
    #    print(output2)
    #    print("\n")
    #    print(ids_to_text(output2, tokenizer))
    #
    # print("-------------")

    torch.manual_seed(seed)
    output3 = generate_loop_kv_cache(
        input_tensor=text_to_ids(prompt, tokenizer=tokenizer),
        model=target_model,
        max_gen=max_gen,
        context_length=draft_context_length,
        top_k=top_k,
        top_p=top_p,
        temp=temp,
        eos_id=50256,
        device=device,
    )
    print("\nOutput from target model:")
    print(output3)
    print("\n")
    print(ids_to_text(output3, tokenizer))
