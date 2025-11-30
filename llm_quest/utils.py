import functools
import math
import re
import time

import numpy as np
import torch
import torch.nn.functional as F


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
        entry (dict): A dictionary containing "question" and "answer" keys from the GSM8K dataset
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
        reasoning_part, _, answer_part = entry["answer"].partition("\n#### ")  # yes 4x "#" from GSM8K
        response_formatted = f"<think>{reasoning_part}</think> <answer>{answer_part}</answer>"

        # we keep "### Response" for now it will be used as separator in ReasoningDataset and added to the prompt
        response = (
            "\n\n### Response:"  
            f"\n{response_formatted}"
            if entry["answer"]

            else ""
        )
        # fmt: on
        return instruction + input_txt + response


class ResponseExtractor:
    """
    Class methods using regex to find content in the response.
    """

    # precompiling regex patterns for efficiency
    # re.DOTALL = not to stop at the end of a line, matches newlines as well
    REASONING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    NUMBER_PATTERN = re.compile(r"[-+]?\s*\d*\.?\d+")
    THOUSAND_SEP_PATTERN = re.compile(r"[,.](?=\d{3})")

    @classmethod
    def get_reasoning(cls, response):
        """
        Extracts the reasoning content from <think> tags in the response.

        Args:
            response (str): The response text containing <think> tags

        Returns:
            str: The reasoning content, or None if not found
        """
        matches = re.findall(cls.REASONING_PATTERN, response)

        if matches:
            return matches[-1].strip()  # strip whitespace and return the reasoning content
        return None

    @classmethod
    def get_answer(cls, response):
        """
        Extracts the final answer content from <answer> tags in the response.
        Ensures the answer is extracted from after the last </think> tag if present.

        Args:
            response (str): The response text that should contain <answer> tags

        Returns:
            str: The answer content (exactly as it appears in the <answer> tags, not sanitized), or None if not found
        """
        # we want to look for an answer after the CoT thinking block (specifically after the last </think> tag seen)
        if "</think>" not in response:  # if there isn't event a think tag, it's not even a valid response
            return None
        response = response.rsplit("</think>", 1)[-1]

        matches = re.findall(cls.ANSWER_PATTERN, response)
        if matches:
            return matches[-1]  # return the latest flagged answer content
        return None

    @classmethod
    def sanitize_answer(cls, answer):
        """
        Sanitizes the answer by removing whitespace, special characters and potential edge cases...
        """
        if not answer:
            return None

        sanitized_answer = answer.strip()
        # handle American (1,000.50) and European (1.000,50) number formats
        sanitized_answer = re.sub(cls.THOUSAND_SEP_PATTERN, "", sanitized_answer)
        sanitized_answer = sanitized_answer.replace(",", ".")  # after removing separators, normalize decimal to dot

        number_match = re.search(cls.NUMBER_PATTERN, sanitized_answer)  # extract the first valid float/int
        if number_match:
            return number_match.group(0).replace(" ", "")  # remove internal spaces ex: "- 72"

        return None


class EntropyFilteredTokens:
    """
    Filters token positions into 3 (non-mutually-exclusive) categories based on the prediction of the next token.
    The filtering method is based on the entropy of the top-k predicted tokens' distribution.
    see 3.3 Pre-Training Setup and 4.1 Language Modeling in the RPT paper

    Args:
        top_k (int): number of top-k tokens to consider for the entropy calculation.
        low (float): entropy threshold for easy tokens.
        mid (float): entropy threshold for medium tokens.
        high (float): entropy threshold for hard tokens.
        pad_token (int): pad token id.

    NOTE: The dataset shouldn't be shuffled, otherwise the global sample indices will be incorrect.
    There's no need to shuffle to filter tokens as a preprocessing step anyway.
    """

    def __init__(self, top_k=16, low=0.5, mid=1, high=1.5, pad_token=50256):
        self.top_k = top_k
        self.hard_indices = []
        self.medium_indices = []
        self.easy_indices = []
        self.pad_token = pad_token

        self.threshold = {
            "hard": high,
            "medium": mid,
            "easy": low,
        }

    @torch.no_grad()
    def process_batch(self, logits, input_ids, global_sample_indices):
        """
        Args:
            logits (torch.Tensor): shape: (batch_size, seq_len, vocab_size)
            input_ids (torch.Tensor): shape: (batch_size, seq_len)
            global_sample_indices (list): list of global sample indices (batch_size,).
                                        Either retrieved from the dataset class or inferred during the training loop:
                                        total samples split in nested lists of len batch_size.
        """
        global_sample_indices = torch.tensor(global_sample_indices, device=logits.device)
        # token i's entropy is for the distribution of token i+1. last real token is meaningful for EoS prediction.
        not_pad_mask = input_ids != self.pad_token

        top_k_logits, _ = torch.topk(logits, self.top_k, dim=-1)
        topk_probas = F.softmax(top_k_logits, dim=-1)
        entropy = -torch.sum(topk_probas * torch.log(topk_probas), dim=-1)  # Shannon entropy of each topk distribution

        for difficulty, threshold in self.threshold.items():
            mask = (entropy > threshold) & not_pad_mask  # shape: (batch_size, seq_len)
            batch_idx, token_idx = torch.where(mask)

            if len(batch_idx) > 0:
                # use local batch_idx to look up the global sample indices
                global_sample_idx = global_sample_indices[batch_idx]
                pair = torch.stack([global_sample_idx, token_idx], dim=1)  # shape: (num_tokens, 2)

                if difficulty == "hard":
                    self.hard_indices.extend(pair.tolist())
                elif difficulty == "medium":
                    self.medium_indices.extend(pair.tolist())
                elif difficulty == "easy":
                    self.easy_indices.extend(pair.tolist())

    def get_difficulty_indices(self):
        """
        Returns:
            dict: A dictionary containing the hard, medium, and easy indices lists where each element is a tuple
            (sample_idx, token_idx)
        """
        return {
            "hard": self.hard_indices,
            "medium": self.medium_indices,
            "easy": self.easy_indices,
        }


class CheckpointEvaluator:
    """
    Evaluator class to check for the best checkpoint in order to save it.
    works with:
    - RLHF GRPO training
    - RLHF Reward Model (RM) training
    - RLVR GRPO training
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
    def is_rlvr_grpo_best(self, kl_div, reward):
        if kl_div > self.kl_div_threshold or reward < self.min_reward_threshold:
            return False

        score = reward - (self.beta * kl_div)
        if score > self.max_score_grpo:
            print(f"New max score found {score:.3f} - saving checkpoint")
            self.max_score_grpo = score
            return True

        return False


# Optimizing KVcache:
# First implementation of KVcache, was concatenating to the existing cache, at each step
# Old KVcache ref: https://github.com/casinca/LLM-quest/commit/0cbf60a078560e77e96cd0ef9a16803f7e5e3240
# then we replaced concats, with indexing inside a pre-allocated cache of length: context_len
#
# 3rd update ref: https://github.com/casinca/LLM-quest/commit/85ac4b14950307c74006e6199fd815c49fe172ae
# optimizing pre-allocated cache by only extending when needed by chunk_size
class KVCache:
    """
    KV cache with dynamic size (increased by chunk_size as sequence length increases) and updating by directly indexing
    into the cache. We avoid torch.cat() operations from the old KVcache.

    Args:
        num_layers (int): Number of transformer layers
        context_len (int): Maximum context_length (max sequence length)
        prompt_len (int): Length of the longest sequence in the batch (prompt length)
        initial_chunk_size (int): Initial extra capacity beyond prompt_len (default: 512)
        chunk_size (int): Size of chunks used to extend the KV cache (default: 256)
    """

    def __init__(self, num_layers, prompt_len, context_len, initial_chunk_size=512, chunk_size=256):
        self.num_layers = num_layers
        self.prompt_len = prompt_len
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.kv_capacity = self.prompt_len + initial_chunk_size

        self.keys_cache = []
        self.values_cache = []

        self.start_pos = 0  # track start of the current sequence length
        self.end_pos = 0  # track end of the current sequence length

    def _initialize(self, batch_size, num_heads, head_dim, device, dtype):
        """
        initialize cache tensors on first call
        """

        for _ in range(self.num_layers):
            self.keys_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    self.kv_capacity,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.values_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    self.kv_capacity,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )

    def _grow_kv_capacity(self, layer_idx):
        """
        Grow the KVcache capacity for a layer, if the total sequence length is greater than the current KV capacity.
        """
        # Only update the (global) KV capacity variable if it hasn't been updated by a previous layer yet
        if self.kv_capacity < self.end_pos:
            if self.end_pos < self.context_len:
                # adding minimum necessary number of chunks. An alt is doubling the chunk size every time
                self.kv_capacity += math.ceil((self.end_pos - self.kv_capacity) / self.chunk_size) * self.chunk_size
            else:
                self.kv_capacity = self.context_len

        # create new extended KVcache for that layer
        new_keys_cache = torch.empty(
            self.batch_size, self.num_heads, self.kv_capacity, self.head_dim, device=self.device, dtype=self.dtype
        )
        new_values_cache = torch.empty(
            self.batch_size, self.num_heads, self.kv_capacity, self.head_dim, device=self.device, dtype=self.dtype
        )

        # copy the old KVcache tensors (up to start_pos) from that layer to the new extended KVcache
        new_keys_cache[:, :, : self.start_pos, :] = self.keys_cache[layer_idx][:, :, : self.start_pos, :]
        new_values_cache[:, :, : self.start_pos, :] = self.values_cache[layer_idx][:, :, : self.start_pos, :]

        # replace layer's KVcache with the new extended KVcache tensors
        self.keys_cache[layer_idx] = new_keys_cache
        self.values_cache[layer_idx] = new_values_cache

    def get_updated_cache(self, keys, values, layer_idx):
        """
        Update cache with the new keys and values and return the full cached keys and values.

        Args:
            keys: New keys tensor of shape (batch_size, num_heads, new_seq_len, head_dim)
            values: New values tensor of shape (batch_size, num_heads, new_seq_len, head_dim)
            layer_idx: Layer index to update

        Note: In most scenarios new_seq_len should be 1

        Returns:
            A tuple containing the full cached keys and values up to the current sequence length.
        """
        self.batch_size, self.num_heads, new_seq_len, self.head_dim = keys.shape
        self.device, self.dtype = keys.device, keys.dtype

        if not self.keys_cache and not self.values_cache:
            self._initialize(self.batch_size, self.num_heads, self.head_dim, self.device, self.dtype)

        self.end_pos = self.start_pos + new_seq_len

        # check end position against the current layer's KV capacity
        curr_layer_kv_capacity = self.keys_cache[layer_idx].shape[2]
        if self.end_pos > curr_layer_kv_capacity:
            self._grow_kv_capacity(layer_idx)

        # update from the new keys and values into the pre-allocated cache
        self.keys_cache[layer_idx][:, :, self.start_pos : self.end_pos, :] = keys
        self.values_cache[layer_idx][:, :, self.start_pos : self.end_pos, :] = values

        # update sequence length after the last layer has been processed for the next call
        if layer_idx == self.num_layers - 1:
            self.start_pos += new_seq_len

        # return slices of the cache, up to the new current sequence length
        return (
            self.keys_cache[layer_idx][:, :, : self.end_pos, :],
            self.values_cache[layer_idx][:, :, : self.end_pos, :],
        )
