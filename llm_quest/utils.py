import functools
import time

import numpy as np
import torch


# timing decorator
def time_it(func):
    """Prints the execution time of the decorated function."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def text_to_ids(text, tokenizer):
    """
    Converts a given text into a tensor of token IDs using a specified tokenizer.

    Args:
        text (str): The input text to be converted into token IDs.
        tokenizer (Tokenizer): The tokenizer object used to convert the text into token IDs.

    Returns:
        torch.Tensor: A tensor containing the token IDs of the input text, with an added batch dimension.
    """

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # convert to tensor and increase dim to fit batch dim format, shape (1, seq_len)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor


def ids_to_text(ids, tokenizer):
    """
    Converts a tensor of token IDs back into text using a specified tokenizer.

    Args:
        ids (torch.Tensor): The tensor containing the token IDs to be converted back into text.
        tokenizer (Tokenizer): The tokenizer object used to convert the token IDs back into text.

    Returns:
        str: The decoded text from the input token IDs.
    """
    encoded_list = ids.squeeze(0).tolist()
    decoded = tokenizer.decode(encoded_list)

    return decoded


def alpaca_prompt_format(entry, include_output=True):
    """
    Formats an instruction-input-output entry into the Alpaca prompt format.

    Args:
        entry (dict): A dictionary containing 'instruction', 'input', and 'output' keys
                    representing an instruction example.
        include_output (bool): If set to False, will remove output_txt from the output

    Returns:
        str: A formatted prompt string containing the instruction, optional input, and output
            in the Alpaca format.
    """

    instruction_txt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        "\n\n### Instruction:"
        f"\n{entry['instruction']}"
    )

    # fmt: off
    input_txt = (
        "\n\n### Input:"
        f"\n{entry['input']}"
        
        if entry["input"]
        else ""
    )

    # small optimization to avoid processing output if not needed
    if not include_output:
        return instruction_txt + input_txt
    
    else:
        output_txt = (
            "\n\n### Response:"
            f"\n{entry['output']}"

            if entry["output"]
            else ""
        )

        return instruction_txt + input_txt + output_txt


# @rasbt's CH5 copy, util function for OpenAI's weights matching
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))


# @rasbt's CH5 copy, util function for OpenAI weight loading (edited for var names matching)
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb_dict.weight = assign(gpt.pos_emb_dict.weight, params["wpe"])
    gpt.emb_dict.weight = assign(gpt.emb_dict.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_queries.weight = assign(gpt.trf_blocks[b].att.w_queries.weight, q_w.T)
        gpt.trf_blocks[b].att.w_keys.weight = assign(gpt.trf_blocks[b].att.w_keys.weight, k_w.T)
        gpt.trf_blocks[b].att.w_values.weight = assign(gpt.trf_blocks[b].att.w_values.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.w_queries.bias = assign(gpt.trf_blocks[b].att.w_queries.bias, q_b)
        gpt.trf_blocks[b].att.w_keys.bias = assign(gpt.trf_blocks[b].att.w_keys.bias, k_b)
        gpt.trf_blocks[b].att.w_values.bias = assign(gpt.trf_blocks[b].att.w_values.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ffn.layers[0].weight = assign(
            gpt.trf_blocks[b].ffn.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ffn.layers[0].bias = assign(
            gpt.trf_blocks[b].ffn.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ffn.layers[2].weight = assign(
            gpt.trf_blocks[b].ffn.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ffn.layers[2].bias = assign(
            gpt.trf_blocks[b].ffn.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ln_1.scale = assign(gpt.trf_blocks[b].ln_1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].ln_1.shift = assign(gpt.trf_blocks[b].ln_1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].ln_2.scale = assign(gpt.trf_blocks[b].ln_2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].ln_2.shift = assign(gpt.trf_blocks[b].ln_2.shift, params["blocks"][b]["ln_2"]["b"])

    gpt.final_ln.scale = assign(gpt.final_ln.scale, params["g"])
    gpt.final_ln.shift = assign(gpt.final_ln.shift, params["b"])
    gpt.out.weight = assign(gpt.out.weight, params["wte"])
