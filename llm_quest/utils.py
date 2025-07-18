import functools
import re
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
        return instruction_txt + input_txt + "\n\n### Response:\n"

    else:
        output_txt = (
            "\n\n### Response:"
            f"\n{entry['output']}"

            if entry["output"]
            else ""
        )

        return instruction_txt + input_txt + output_txt


def alpaca_deepseek_format(entry, include_response=True):
    """
    Formats a GSM8K entry into a DeepSeek reasoning format (R1 paper) adapted with alpaca style instruction:
    - includes alpaca style instruction
    - includes reasoning and final answer tags for the answer


    Args:
        entry (dict): A dictionary containing 'question' and 'answer' keys
                        representing a math problem example with reasoning and final answer.
        include_response (bool): If set to False, will remove the formatted response (reasoning+answer) from the output.

    Returns:
        str: A formatted prompt string containing the question and, optionally, the structured answer following DeepSeek
        R1 paper format (with <think> and <answer> tags) adapted to the alpaca style instruction.
    """

    # TODO: test some different good instructs that aren't too long and complex + close semantically to alpaca SFT and
    # see if it improves the performance
    instruction = (
        "### Instruction:\n"
        "Below is a question concerning a math problem. "
        "Your role as an assistant is to reason step by step and provide the final answer to the problem. "
        "It is very important that you structure your response into 2 main sections: reasoning and answer. "
        "You must enclose your reasoning process in <think> </think> tags and final answer in <answer> </answer> tags. "
        "For example: <think> reasoning process here </think> <answer> answer here </answer>. "
        "Following the above instructions, try to solve the question:"
    )

    # fmt: off
    input_txt = (
        "\n\n### Input:"
        f"\n{entry['question']}"
        
        if entry["question"]
        else ""
    )

    # small optimization to avoid processing response if not needed
    if not include_response:
        return instruction + input_txt

    else:
        reasoning_part, separator, answer_part = entry["answer"].partition("\n#### ")
        response_formatted = f"<think>{reasoning_part}</think> <answer>{answer_part}</answer>"
        
        response = (
            "\n\n### Response:"
            f"\n{response_formatted}"

            if entry["answer"]
            else ""
        )

        return instruction +input_txt + response


class ResponseExtractor:
    """
    Static functions using regex to find content in the response.
    """

    @staticmethod
    def get_reasoning(response):
        """
        Extracts the reasoning content from <think> tags in the response.

        Args:
            response (str): The response text containing <think> tags

        Returns:
            str: The reasoning content, or None if not found
        """

        # important re.DOTALL not to stop at the end of a line, match newlines as well
        matches = re.findall(r"<think>(.*?)</think>", response, re.DOTALL)

        if matches:
            return matches[-1].strip()  # strip whitespace and return the reasoning content
        return None

    @staticmethod
    def get_answer(response):
        """
        Extracts the final answer content from <answer> tags in the response.

        Args:
            response (str): The response text containing <answer> tags

        Returns:
            str: The answer content, or None if not found
        """

        # important re.DOTALL not to stop at the end of a line, match newlines as well
        matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if matches:
            return matches[-1]  # strip whitespace and return the answer content
        return None

    @staticmethod
    def sanitize_answer(answer):
        """
        Sanitizes the answer by removing whitespace, special characters and potential edge cases...
        """
        if not answer:
            return None

        sanitized_answer = answer.strip()
        # handle American (1,000.50) and European (1.000,50) number formats
        sanitized_answer = re.sub(r",(?=\d{3})", "", sanitized_answer)  # For 1,000
        sanitized_answer = re.sub(r"\.(?=\d{3})", "", sanitized_answer)  # For 1.000
        sanitized_answer = sanitized_answer.replace(",", ".")  # after removing separators, normalize decimal to dot

        number_match = re.search(r"[-+]?\s*\d*\.?\d+", sanitized_answer)  # extract the first valid float/int
        if number_match:
            return number_match.group(0).replace(" ", "")  # remove internal spaces ex: "- 72"

        return None


class CheckpointEvaluator:
    """
    Evaluator class to check for the best checkpoint in order to save it.
    works with:
    - RLHF GRPO training
    - RLHF Reward Model (RM) training
    """

    def __init__(
        self,
        kl_div_threshold=0.5,
        min_reward_threshold=6.0,
        beta=1.0,
        rm_min_accuracy_threshold=0.9,
        rm_min_val_loss_threshold=0.1,
    ):
        """
        Args:
            kl_div_threshold (float): minimum kl div threshold to save a checkpoint during RLHF GRPO training
            min_reward_threshold (float): minimum reward score threshold to save a checkpoint during RLHF GRPO
            training
            beta (float): coeff for the KL div penalty in RLHF GRPO training
            rm_min_accuracy_threshold (float): minimum accuracy threshold to save a checkpoint during RLHF RM training
            rm_min_val_loss_threshold (float): minimum validation loss threshold to save a checkpoint during RLHF RM
            training
        """
        self.kl_div_threshold = kl_div_threshold
        self.min_reward_threshold = min_reward_threshold
        self.beta = beta
        self.max_score_grpo = float("-inf")
        self.max_accu_pref_rm = float("-inf")
        self.rm_min_accuracy_threshold = rm_min_accuracy_threshold
        self.rm_min_val_loss_threshold = rm_min_val_loss_threshold

    def is_rlhf_grpo_best(self, kl_div, reward):
        """
        Method of eval: Simple KL div penalized reward.
        """
        if kl_div > self.kl_div_threshold or reward < self.min_reward_threshold:
            return False

        score = reward - (self.beta * kl_div)
        if score > self.max_score_grpo:
            print(f"New max score found {score:.3f} - saving checkpoint")
            self.max_score_grpo = score
            return True

        return False

    def is_rm_accu_best(self, accuracy, val_loss):
        if accuracy < self.rm_min_accuracy_threshold or val_loss > self.rm_min_val_loss_threshold:
            return False

        if accuracy > self.max_accu_pref_rm:
            print(f"New max accuracy found {accuracy:.3f} - saving checkpoint")
            self.max_accu_pref_rm = accuracy
            return True
        return False

    # same as rlhf grpo logic but separate in case we need to change the logic
    def is_rlvr_grpo_best(self, reward, kl_div):
        if kl_div > self.kl_div_threshold or reward < self.min_reward_threshold:
            return False

        score = reward - (self.beta * kl_div)
        if score > self.max_score_grpo:
            print(f"New max score found {score:.3f} - saving checkpoint")
            self.max_score_grpo = score
            return True

        return False


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
