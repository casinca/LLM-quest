import json

import tiktoken
import torch

import config
from llm_quest.generate import generate_loop
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import alpaca_prompt_format, ids_to_text, text_to_ids

model_device = config.auto_device
EOS_ID = 50256


def generate_response(model, input_text):
    """Quick wrapper for generate_loop"""
    token_ids = text_to_ids(input_text, tokenizer).to(model_device)
    output_ids = generate_loop(
        token_ids,
        model,
        max_gen=35,
        context_length=model_cfg["context_length"],
        temp=1,
        top_k=20,
        eos_id=EOS_ID,
        device=model_device,
    )
    generated_text = ids_to_text(output_ids, tokenizer)
    return generated_text[len(input_text) :].replace("### Response:", "").strip()


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(config.instruct_preference_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)[55:58]  # same samples that were used in llm-from-scratch repo for DPO comparison

    # --- Model Config and Loading ---
    model_cfg = config.gpt2_config_creator("gpt_m")

    policy_model = GPTModel(model_cfg)
    policy_checkpoint = torch.load(
        config.rlhf_grpo_checkpoint_dir / "best_checkpoint_150_score_6.370_gpro_loss_fix.pt",
        map_location=model_device,
        weights_only=True,
    )
    policy_model.load_state_dict(policy_checkpoint)
    policy_model.to(model_device)
    policy_model.eval()

    reference_model = GPTModel(model_cfg)
    ref_checkpoint = torch.load(config.ft_instruct_w_gpt2, map_location=model_device, weights_only=True)
    reference_model.load_state_dict(ref_checkpoint["model_state_dict"])
    reference_model.to(model_device)
    reference_model.eval()

    # --- Generation ---
    for entry in test_data:
        input_text = alpaca_prompt_format(entry, include_output=False)

        reference_response_text = generate_response(reference_model, input_text)
        policy_response_text = generate_response(policy_model, input_text)

        print(input_text.replace("\n\n### Response:", "").strip())
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nReference model response:\n>> {reference_response_text}")
        print(f"\nPolicy model response:\n>> {policy_response_text}")
        print("\n" + "-" * 50 + "\n")
