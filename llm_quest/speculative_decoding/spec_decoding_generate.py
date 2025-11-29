import tiktoken
import torch

import config
from llm_quest.generate import generate_loop_kv_cache
from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.speculative_decoding.spec_decoding_engine import speculative_generate
from llm_quest.utils import ids_to_text, text_to_ids

# --- Hyperparameters ---
max_gen = 200
draft_max_gen = 5  # gamma(γ) in the paper
seed = 123
temp = 0.0
top_k = None
top_p = None

prompt = "All there is"

if __name__ == "__main__":

    target_weights_path = download_gpt_model(gpt_size="gpt_l", save_dir=config.openai_pretrained_w_gpt2_l)
    draft_weights_path = download_gpt_model(gpt_size="gpt_s", save_dir=config.openai_pretrained_w_gpt2_s)

    tokenizer = tiktoken.get_encoding("gpt2")

    target_model_config = config.gpt2_config_creator("gpt_l")  # needs a larger model to make speculative decoding worth
    draft_model_config = config.gpt2_config_creator("gpt_s")

    device = config.auto_device

    target_model = GPTModel(target_model_config)
    draft_model = GPTModel(draft_model_config)

    load_gpt_weights(target_model, target_weights_path)
    load_gpt_weights(draft_model, draft_weights_path)

    target_model.to(device=device, dtype=torch.float32).eval()
    draft_model.to(device=device, dtype=torch.float32).eval()

    torch.manual_seed(seed)
    output = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt=text_to_ids(prompt, tokenizer=tokenizer),
        max_gen=max_gen,
        context_length=draft_model_config["context_length"],
        draft_max_gen=draft_max_gen,
        top_k=top_k,
        top_p=top_p,
        temp=temp,
        eos_id=50256,
        device=device,
    )

    print("\nOutput from speculative decoding:")
    print(output)
    print(output.shape)
    print("\n")
    print(ids_to_text(output, tokenizer))

    print("-------------")

    torch.manual_seed(seed)
    output2 = generate_loop_kv_cache(
        input_tensor=text_to_ids(prompt, tokenizer=tokenizer),
        model=target_model,
        max_gen=max_gen,
        context_length=target_model_config["context_length"],
        top_k=top_k,
        top_p=top_p,
        temp=temp,
        eos_ids=50256,
        device=device,
    )

    print("\nOutput from target model:")
    print(output2)
    print(output2.shape)
    print("\n")
    print(ids_to_text(output2, tokenizer))
