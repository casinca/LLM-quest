import functools
import itertools
import math
import re
import time

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


# NOTE:
# - since we check for "all" matrices convergence, one outlier/slow convergence matrix will force extra iterations
# for the whole batch
# - probably some other optimizations log
# TODO: check for a good iter_check, clamp values and training stability
class SinkhornKnopp:
    """
    Sinkhorn-Knopp Algorithm implemented in PyTorch.
    This implementation is adapted from this NumPy version by @btaba: https://github.com/btaba/sinkhorn_knopp
    Original paper: http://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-s.pdf

    2 changes made:
    - Supports batch dimensions >=2D (e.g., (N, N), (B, N, N), (B, S, N, N) ...).
        Operations are performed on the last two dimensions.
    - Optional check for convergence every "iter_check" iterations

    Takes a non-negative square matrix P, where P =/= 0 and iterates through
    Sinkhorn Knopp's algorithm to convert P to a doubly stochastic matrix.
    Guaranteed convergence if P has total support.

    Parameters
    ----------
    max_iter : int, default=20
        The maximum number of iterations.
    epsilon : float, default=1e-6
        Convergence threshold. Algorithm stops when all row and column sums
        are within epsilon of 1. Must be between 0 and 1.
    iter_check : int, default=3
        Check convergence every iter_check iterations, set to 0 to check every iteration.

    Attributes
    ----------
    iterations : int
        Number of iterations from the last call.
    stopping_condition : str
        Either "max_iter" or "epsilon", describing why the algorithm stopped.

    Examples
    --------
    >>> P = torch.tensor([[0.011, 0.15], [1.71, 0.1]])
    >>> sk = SinkhornKnopp(max_iter=20, epsilon=1e-6)
    >>> P_ds = sk(P)
    >>> print(P_ds)
    >>> print(P_ds.sum(dim=-1))  # Row sums ≈ 1
    >>> print(P_ds.sum(dim=-2))  # Column sums ≈ 1

    >>> # 4D tensor example (batch, seq_len, N , N)
    >>> P_4d = torch.rand(2, 10, 4, 4) + 0.1
    >>> P_ds_4d = sk(P_4d)
    """

    def __init__(self, max_iter=20, epsilon=1e-6, iter_check=3):
        assert max_iter > 0, f"max_iter must be positive, got {max_iter}"
        assert 0 < epsilon < 1, f"epsilon must be in (0, 1), got {epsilon}"

        self.max_iter = max_iter
        self.epsilon = epsilon
        self.iter_check = iter_check
        self.iterations = 0
        self.stopping_condition = None
        self._r = None
        self._c = None

    def __call__(self, P):
        """
        Transform a matrix (or batch of matrices) to doubly stochastic form

        Parameters
        ----------
        P : torch.Tensor
            Non-negative square matrix of shape >=2D (..., N, N).


        Returns
        -------
        torch.Tensor
            Doubly stochastic matrix (or batch) of the same shape as input.
        """
        return self.fit(P)

    def fit(self, P):
        """
        Fit and transform a matrix (or batch of matrices) to doubly stochastic form.

        Parameters
        ----------
        P : torch.Tensor
            Non-negative square matrix of shape >=2D (..., N, N).
            Operations are performed on the last two dimensions.

        Returns
        -------
        torch.Tensor
            Doubly stochastic matrix (or batch of matrices) of the same shape as input.
        """
        assert P.dim() >= 2, f"Expected at least 2D tensor, got {P.dim()}D"

        N, M = P.shape[-2:]  # get matrix dimensions from last two dims
        assert N == M, f"Matrix must be square, got shape ({N}, {M})"
        assert (P >= 0).all(), "Matrix must be non-negative"

        min_thresh = 1.0 - self.epsilon
        max_thresh = 1.0 + self.epsilon

        # Save original shape and flatten leading dimensions (bs, N, N) with bs = product of all leading dims
        original_shape = P.shape
        P_2d = P.view(-1, N, M)
        bs = P_2d.shape[0]

        # Initialize scaling vectors for each matrix in the batch
        # r scales rows: (bs, N, 1)
        # c scales columns: (bs, 1, M)
        r = torch.ones(bs, N, 1, dtype=P.dtype, device=P.device)
        c = torch.ones(bs, 1, M, dtype=P.dtype, device=P.device)

        self.iterations = 0
        self.stopping_condition = None

        for self.iterations in range(1, self.max_iter + 1):
            # Column normalization: c_j = 1 / sum_i(r_i * P_ij)
            # Sum over rows (dim=1) gives shape (bs, 1, M)
            c = 1.0 / (P_2d * r).sum(dim=1, keepdim=True).clamp(min=1e-10)

            # Row normalization: r_i = 1 / sum_j(P_ij * c_j)
            # Sum over columns (dim=2) gives shape (bs, N, 1)
            r = 1.0 / (P_2d * c).sum(dim=2, keepdim=True).clamp(min=1e-10)

            # Check convergence every "iter_check" iterations
            if self.iterations % self.iter_check == 0:
                P_scaled = r * P_2d * c
                row_sums = P_scaled.sum(dim=-1)  # (bs, N)
                col_sums = P_scaled.sum(dim=-2)  # (bs, M)

                row_converged = (row_sums >= min_thresh) & (row_sums <= max_thresh)
                col_converged = (col_sums >= min_thresh) & (col_sums <= max_thresh)

                if row_converged.all() and col_converged.all():
                    self.stopping_condition = "epsilon"
                    break

        if self.stopping_condition is None:
            self.stopping_condition = "max_iter"

        self._r = r
        self._c = c

        # final scaled matrix and reshape back to original shape (bs, N, N) → (..., N, N)
        P_ds = r * P_2d * c
        return P_ds.view(original_shape)


class BirkhoffvonNeumann(torch.nn.Module):
    """
    This class is used to build the doubly stochastic H_res matrix for the DeepSeek mHC optimization:
    "mHC-lite"

    We are building H_res as a convex combination/weighted average from the Birkhoff–von Neumann
    decomposition/theorem depicted in p.4 of the mHC-lite paper https://arxiv.org/abs/2601.05732

    H_res = sum(a_k * P_k)

    with:
        k from 1 to n!
        n is the expansion rate in mHC and mHC-lite
        P_k as a permutation matrix (n x n)
        a_k is a weight scalar from the `weight_a` softmaxed vector (each scalar is >0 and the total sum to 1)

    """

    def __init__(self, expansion_rate):
        """
        Args:
            expansion_rate (int): The number of expanded streams in the case of DeepSeek mHC/mHC-lite
                                This is the dimension n of the (n x n) square matrices P_k in the mHC-lite paper
        """
        super().__init__()
        # since expansion_rate is small from HC and mHC, we are caching the permutation matrices for efficiency
        assert expansion_rate <= 8, "Expansion rate must be <= 8 to avoid memory issues, with more than 8! matrices"

        self.num_permut = math.factorial(expansion_rate)
        self.exps_rate = expansion_rate
        self.permutations = list(itertools.permutations(range(self.exps_rate)))
        self.identity_permut_index = self._get_identity_permutation_index()

        flat_permut_matrices = self._get_permut_matrices(dtype=torch.float32)  # TODO for now we dtype .to(weight_a)
        self.register_buffer("flat_permut_matrices", flat_permut_matrices, persistent=False)

    def _get_identity_permutation_index(self):
        """
        Returns the index (int) of the identity permutation in the list of permutations
        """
        # The identity permutation should be the first one (index 0) since itertools.permutations should always return
        # the range(n) as the first permutation but for safety we can iter anyway
        identity_perm = tuple(range(self.exps_rate))
        try:
            identity_index = self.permutations.index(identity_perm)
        except ValueError:
            raise ValueError("Identity permutation not found")  #  shouldn't happen though

        return identity_index

    def _get_permut_matrices(self, dtype=None):
        """
        Get the flattened permutation matrices P_k

        We are not returning the permutation matrices as shape (n!, n, n) because we will be doing a matmul, for
        efficiency, to compute the convex combination/ weighted average sum(a_k * P_k) to get H_res in the forward
        (bvn_composition) method.

        Args:
            dtype (torch.dtype): The dtype of the permutation matrices, default: None

        Returns:
            The flattened permutation matrices P_k, shape: (n!, n * n)
        """
        # we create permutation matrices P_k by re-arranging rows of the identity matrices created from torch.eye()
        indices = torch.tensor(self.permutations, dtype=torch.long)
        # same as doing:
        # self.permut_matrices = F.one_hot(indices, num_classes=self.exps_rate).to(dtype=dtype, device=device)
        permut_matrices = torch.eye(self.exps_rate, dtype=dtype)[indices]
        # flatten P_k matrices to list of vectors (n!, n, n) → (n!, n*n)
        return permut_matrices.view(self.num_permut, -1)

    def bvn_composition(self, weight_a):
        """
        Compose H_res from the Birkhoff–von Neumann theorem H_res = sum(a_k * P_k) with k from 1 to n!
        See Theorem 3.1 p.4 in the mHC-lite paper

        For efficiency, we are doing a weighted average of the permutation matrices P_k with the weights in `weight_a`,
        computed as a matmul.

        For example n = 2, flattened and stacked P_k matrices:

            self.permut_matrices_flat = [[ --P_1(flat)-- ]
                                        [ --P_2(flat)-- ]]

        `weight_a` vector [a, b], which contains the weights for each permutation, we compute the convex combination as
        a vector matrix product
        (vm product here for a single `weight_a`, otherwise as a matmul with multiple seq_len `weight_a` vectors)

            [a, b] x [[ --P1(flat)-- ]
                    [ --P2(flat)-- ]]

        which gives a vector of shape (n*n) [a*P1_1 + b*P2_1, a*P1_2 + b*P2_2, ...]
        reshaped back to (n, n) to get H_res as a doubly stochastic matrix.

        Args:
            weight_a (torch.Tensor): A weight_a vector for each token, with scalar weights for each of the permutation
            matrices, shape: (b, seq_len, n!)
            weight_a scalars must be >=0 and sum to 1 (this is done via softmax in MHCLiteRes class)

        Returns:
            The bi/doubly stochastic matrix H_res for each token, shape: (b, seq_len, n, n)
        """
        b, seq_len, _ = weight_a.shape

        H_res = weight_a @ self.flat_permut_matrices.to(weight_a.dtype)
        H_res = H_res.view(b, seq_len, self.exps_rate, self.exps_rate)

        return H_res

    def __call__(self, weight_a):
        return self.bvn_composition(weight_a)
