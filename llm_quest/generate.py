import torch

from llm_quest.gpt.gpt_attention import KVCache


def generate_simple_loop(input_tensor, model, max_gen, context_length):
    """
    simplified generate_loop function
    """
    for i in range(max_gen):
        # truncate token ids to compatible context size, shape (b, seq_len) → (b, ctx_len)
        trunc_input = input_tensor[:, -context_length:]

        with torch.inference_mode():  # no need for grad as we're generating
            logits = model(trunc_input)
        # taking last vector since goal is "next word" prediction, and getting idx of the highest logit
        logits = logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # keepdim to concat (needs same dim)
        input_tensor = torch.cat(
            (input_tensor, next_token), dim=-1
        )  # adding predicted token id back to the input for the next loop

    # final "input" is actually initial input+all predicted token ids
    return input_tensor


def generate_loop(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    top_p=None,
    min_p=None,
    temp=0.0,
    eos_id=None,
    device=torch.device("cuda"),
):
    """
    Generates text using a GPT model with optional top-k sampling, temperature scaling, and early stopping.

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs.
        model (nn.Module): The GPT model used for text generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        top_p (float, optional): If specified, limits sampling to top p most likely tokens. Can be combined with top_k.
                                Defaults to None.
        min_p (float, optional): If specified, limits sampling to tokens based on a dynamic threshold (scaled by the
        probability of the most likely token.)
                                Defaults to None
        temp (float, optional): Temperature sampling:
                                - if >1, increases entropy (randomness)
                                - if <1, decreases entropy (more deterministic)
                                - if 1, untempered distribution
                                - if 0, uses greedy sampling. Defaults to 0.0.

        eos_id (int, optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to None.
        device (torch.device or str, optional): Device to move the input tensor to. Defaults to "cuda".

    Returns:
        torch.Tensor: Input tensor concatenated with generated token IDs
    """
    input_tensor = input_tensor.to(device)

    for i in range(max_gen):
        # truncate input to compatible context size, shape (b, ctx_len)
        trunc_input = input_tensor[:, -context_length:]

        with torch.inference_mode():  # no need for grads as we're generating
            logits = model(trunc_input)[:, -1, :]  # taking last vector (next word prediction)

        next_token = sampling(logits, top_k, top_p, min_p, temp)

        if eos_id is not None and next_token == eos_id:  # if a EoT is seen stops the generation earlier
            break

        input_tensor = torch.cat(
            (input_tensor, next_token), dim=-1
        )  # adding chosen token id back to the input for the next loop

    # final "input" is actually initial input+all predicted words
    return input_tensor


def generate_loop_kv_cache(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    top_p=None,
    min_p=None,
    temp=0.0,
    eos_ids=None,
    device=torch.device("cuda"),
    rope_model=False,
):
    """Standalone function, same as generate_loop() but with KV cache.

    Args:
        rope_model (bool, optional): Whether the model is a RoPE model. Defaults to False.
                                    If True, position_ids are handled differently for RoPE models.
    """

    token_ids = []  # little optim to avoid repeated concat in the loop. store token ids and concat once at the end

    num_layers = len(model.trf_blocks)
    # Init KV cache
    kv_cache = KVCache(num_layers=num_layers, context_len=context_length)

    input_tensor = input_tensor.to(device)

    if eos_ids is not None:
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]
        eos_ids_tensor = torch.tensor(eos_ids, device=device, dtype=torch.long)

    # truncate input to compatible context size, shape (b, ctx_len)
    trunc_input = input_tensor[:, -context_length:]
    # For RoPE models when using KVCache and incrementing position_id, shape (batch_size(1), 1)
    next_position_id = torch.tensor([[trunc_input.shape[-1]]], dtype=torch.long, device=device)

    # --- First generation to build the kv cache ---
    with torch.inference_mode():
        logits = model(trunc_input, kv_cache=kv_cache)[:, -1, :]

        # --- Continuing generations with kv cache ---
        for _ in range(max_gen):
            next_token = sampling(logits, top_k, top_p, min_p, temp)

            # if any of the EoS tokens are seen, stop the generation
            if eos_ids is not None and torch.isin(next_token, eos_ids_tensor).any():
                break

            token_ids.append(next_token)

            # since 1 token/seq, we can also squeeze now (b, 1, v) → (b, v) same as logits[:, -1, :]
            if rope_model:
                logits = model(next_token, kv_cache=kv_cache, position_ids=next_position_id).squeeze(1)
                next_position_id += 1
            else:
                logits = model(next_token, kv_cache=kv_cache).squeeze(1)

    # final "input" is actually initial input+all predicted words
    return torch.cat([input_tensor] + token_ids, dim=-1)


# Early version for batching prompts without KVCache and right padding only
#
# dynamic attention_mask: Avoid padding tokens from shorter prompts while generating
# dynamic generation:
#   - only generating for unfinished prompts (more efficient vs continue useless generations after EoS)
#   - early finished generations are out of the loop and padded with eos_id
# retrieving the last real token's prediction for the first step (because of right padding)
def generate_batched_loop(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    top_p=None,
    temp=0.0,
    eos_id=50256,
    device=torch.device("cuda"),
    attention_mask=None,
    last_real=None,
):
    """
    Generates text from batched prompts, handling dynamic attention masks and early stopping for individual sequences.
    We retrieve the last real token's prediction for the first step for right padding case.

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs.
        model (nn.Module): The model used for generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        top_p (float, optional): If specified, limits sampling to top p most likely tokens. Can be combined with top_k.
                                Defaults to None.
        temp (float, optional): Sampling temperature. A higher value makes the output more random.
                                if 1, untempered distribution.
                                Defaults to 0.0 (greedy sampling).
        eos_id (int, optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to 50256 (GPT-2 EOS token).
        device (torch.device or str, optional): The device to perform computations on. Defaults to "cuda".
        attention_mask (torch.Tensor, optional): A boolean tensor of shape (batch_size, sequence_length) indicating
                                                which tokens are real (True) and which are padding (False).
                                                Used during attention calculation. Defaults to None.
        last_real (torch.Tensor, optional): A tensor of shape (batch_size,) indicating the index of the last real token
                                            in each prompt of `input_tensor`. Used in the first generation step to
                                            correctly extract logits for right-padded inputs. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, prompt_length + generated_length) containing the original prompts
                        concatenated with the generated token IDs. Sequences that finished early will be padded with
                        `eos_id` up to `max_gen` length.
    """
    input_tensor = input_tensor.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if last_real is not None:
        last_real = last_real.to(device)

    batch_size = input_tensor.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # tracking if a generation is finished

    for step in range(max_gen):
        if finished.all():  # early exit
            break

        # we build the input only for unfinished rows/sequences
        # for simplicity, calling "N_active" the number of True values in the mask/batch.
        unfinished = ~finished  # bool mask of unfinished generations (batch_size,)
        trunc_input = input_tensor[unfinished, -context_length:]

        curr_mask = None
        if attention_mask is not None:
            curr_mask = attention_mask[unfinished, -context_length:]

        with torch.inference_mode():
            logits = model(trunc_input, attn_mask=curr_mask)  # (N_active, seq_len, v)

        if step == 0 and last_real is not None:
            seq_pos = torch.arange(batch_size, device=device)  # N_active=batch size for the first step
            logits = logits[seq_pos, last_real[unfinished], :]
        else:
            logits = logits[:, -1, :]  # (N_active, v)

        # sampling for the N_active/unfinished rows
        next_toks = sampling(logits, top_k, top_p, temp)  # (N_active, 1)

        # create a tensor full of eos_id, insert sampled tokens back into the batch only for unfinished generations
        full_next_toks = torch.full((batch_size, 1), eos_id, device=device, dtype=torch.long)
        full_next_toks[unfinished] = next_toks

        # append to running tensors and extend attention mask with True for the newly generated token
        input_tensor = torch.cat([input_tensor, full_next_toks], dim=-1)
        if attention_mask is not None:
            new_mask = torch.ones_like(full_next_toks, dtype=torch.bool)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        # update finished flag
        finished |= full_next_toks.squeeze(1) == eos_id

    return input_tensor


def generate_batched_loop_kv_cache(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    top_p=None,
    min_p=None,
    temp=0.0,
    eos_ids=50256,
    pad_id=50256,
    device=torch.device("cuda"),
    last_real=None,
    rope_model=True,
    *,
    attention_mask,  # emphasizing that now is a required argument even for single batch
):
    """
    Generates text from batched prompts using KV cache, handling dynamic attention masks for right padding.
    This is a simplified version of generate_batched_loop() using KVcache from generate_loop_kv_cache().

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs.
        model (nn.Module): The model used for generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        top_p (float, optional): If specified, limits sampling to top p most likely tokens. Can be combined with top_k.
                                Defaults to None.
        min_p (float, optional): If specified, limits sampling to tokens based on a dynamic threshold (scaled by the
        probability of the most likely token.)
                                Defaults to None
        temp (float, optional): Sampling temperature. A higher value makes the output more random.
                                if 1, untempered distribution.
                                Defaults to 0.0 (greedy sampling).
        eos_ids (int | List[int], optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to 50256 (GPT-2 EOS token).
        pad_id (int, optional): Token ID that signals padding.
                                Defaults to 50256 (GPT-2 EOS token).
        device (torch.device or str, optional): The device to perform computations on. Defaults to "cuda".
        attention_mask (torch.Tensor ): A boolean tensor of shape (batch_size, sequence_length) indicating
                                                which tokens are real (True) and which are padding (False).
                                                Used for attention masking + position_ids for RoPE.
        last_real (torch.Tensor, optional): A tensor of shape (batch_size,) indicating the index of the last real token
                                            in each prompt of `input_tensor`. Used for backwards compatibility with
                                            position_ids calculation. Defaults to None.
        rope_model (bool, optional): Backward compatibility for GPT2.
                                    Whether the model is a RoPE model. Defaults to True.
                                    If False, positional information with padding is handled differently.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, prompt_length + generated_length) containing the original prompts
                        concatenated with the generated token IDs.
    """
    input_tensor = input_tensor.to(device)
    attention_mask = attention_mask.bool().to(device)

    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    eos_ids_tensor = torch.tensor(eos_ids, device=device, dtype=torch.long)

    if last_real is not None:
        last_real = last_real.to(device)

    batch_size = input_tensor.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    num_layers = len(model.trf_blocks)
    kv_cache = KVCache(num_layers=num_layers, context_len=context_length)

    generated_tokens = []

    # --- Position IDs handling for right padding ---
    # (use last_real if provided for backwards compatibility)
    if last_real is not None:
        next_pos_ids = last_real.unsqueeze(-1) + 1  # (batch_size, 1)
    else:
        next_pos_ids = attention_mask.sum(dim=-1, keepdim=True)  # (batch_size, 1)
        last_real = next_pos_ids.clone().squeeze(-1) - 1  # (batch_size,)

    # --- First generation to build the kv cache ---
    with torch.inference_mode():
        # whether RoPE model or not, we don't need specific position_ids for the first generation, in right padding
        logits = model(input_tensor, attn_mask=attention_mask, kv_cache=kv_cache)

    # extract logits from the last real token position for right padding
    seq_pos = torch.arange(batch_size, device=device)
    logits = logits[seq_pos, last_real, :]

    next_token = sampling(logits, top_k, top_p, min_p, temp)
    generated_tokens.append(next_token)
    finished |= torch.isin(next_token.squeeze(1), eos_ids_tensor)
    attention_mask = torch.cat([attention_mask, (~finished).unsqueeze(-1)], dim=-1)

    # --- Continuing generations with kv cache ---
    for _ in range(max_gen - 1):
        if finished.all():
            break

        with torch.inference_mode():
            if rope_model:
                logits = model(
                    next_token,
                    attn_mask=attention_mask,
                    kv_cache=kv_cache,
                    position_ids=next_pos_ids,
                ).squeeze(1)
                next_pos_ids += 1
            else:
                logits = model(
                    next_token,
                    attn_mask=attention_mask,
                    kv_cache=kv_cache,
                ).squeeze(1)

        sampled_tokens = sampling(logits, top_k, top_p, min_p, temp)

        # For finished sequences, we keep appending Pad token. For unfinished sequences, we append the new token.
        next_token = torch.where(
            finished.unsqueeze(-1), torch.tensor(pad_id, device=device, dtype=torch.long), sampled_tokens
        )
        generated_tokens.append(next_token)
        finished |= torch.isin(next_token.squeeze(1), eos_ids_tensor)

        # extend attention mask: True for unfinished sequences
        new_mask = (~finished).unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

    all_generated = torch.cat(generated_tokens, dim=1)
    return torch.cat([input_tensor, all_generated], dim=1)


def generate_batched_loop_kv_cache_left_pad(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    top_p=None,
    min_p=None,
    temp=0.0,
    eos_id=50256,
    device=torch.device("cuda"),
    rope_model=True,  # placeholder for API
    *,
    attention_mask,
):
    """
    Generates text from batched prompts with left padding, using KV cache.

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs with left padding.
        model (nn.Module): The model used for generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        top_p (float, optional): If specified, limits sampling to top p most likely tokens. Can be combined with top_k.
                                Defaults to None.
        min_p (float, optional): If specified, limits sampling to tokens based on a dynamic threshold (scaled by the
        probability of the most likely token.)
                                Defaults to None
        temp (float, optional): Sampling temperature. A higher value makes the output more random.
                                if 1, untempered distribution.
                                Defaults to 0.0 (greedy sampling).
        eos_id (int, optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to 50256 (GPT-2 EOS token).
        device (torch.device or str, optional): The device to perform computations on. Defaults to "cuda".
        attention_mask (torch.Tensor): tensor of shape (batch_size, sequence_length) indicating
                                                which tokens are real (True) and which are padding (False).
                                                Used for Attention calc + token positions tracking
    Returns:
        torch.Tensor: A tensor of shape (batch_size, prompt_length + generated_length) containing the original prompts
                        concatenated with the generated token IDs.
    """
    input_tensor = input_tensor.to(device)
    attention_mask = attention_mask.bool().to(device)

    batch_size = input_tensor.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    num_layers = len(model.trf_blocks)
    kv_cache = KVCache(num_layers=num_layers, context_len=context_length)

    # --- Position IDs handling for left padding ---
    # creating position_ids for first forward pass RoPE + left padding, ex: attn [0,0,1,1,1] → pos [0,0,0,1,2]
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(~attention_mask, 0)  # (batch_size, seq_len)
    # for subsequent forward passes, we only need to track the position of the next token to generate, ex [3]
    next_pos_id = attention_mask.sum(dim=-1, keepdim=True)  # (batch_size, 1)

    generated_tokens = []

    # --- First generation to build the kv cache ---
    with torch.inference_mode():
        logits = model(input_tensor, attn_mask=attention_mask, kv_cache=kv_cache, position_ids=position_ids)

    # with left padding, all sequences end at position -1 (no need to track last real token like right padding)
    logits = logits[:, -1, :]

    next_token = sampling(logits, top_k, top_p, min_p, temp)
    generated_tokens.append(next_token)
    finished |= next_token.squeeze(1) == eos_id

    # --- Continuing generations with kv cache ---
    for _ in range(max_gen - 1):
        if finished.all():
            break

        # extend attention mask with True for the newly generated token
        new_mask = torch.ones_like(next_token, dtype=torch.bool)
        attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        with torch.inference_mode():
            logits = model(
                next_token,
                attn_mask=attention_mask,
                kv_cache=kv_cache,
                position_ids=next_pos_id,
            ).squeeze(1)
        next_pos_id += 1

        sampled_tokens = sampling(logits, top_k, top_p, min_p, temp)

        # For finished sequences, we keep appending EoS. For unfinished sequences, we append the new token.
        next_token = torch.where(
            finished.unsqueeze(-1), torch.tensor(eos_id, device=device, dtype=torch.long), sampled_tokens
        )
        generated_tokens.append(next_token)
        finished |= next_token.squeeze(1) == eos_id

    all_generated = torch.cat(generated_tokens, dim=1)
    return torch.cat([input_tensor, all_generated], dim=1)


def sampling(logits, top_k=None, top_p=None, min_p=None, temp=0.0):
    """
    Performs sampling on the logits.

    Args:
        logits (torch.Tensor): logits tensor representing last tokens raw probs (scores) in a batch, shape (b, v)
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        top_p (float, optional): If specified, limits sampling to top p most likely tokens. Can be combined with top_k.
                                Defaults to None, range [0.0, 1.0].
        min_p (float, optional): If specified, limits sampling to tokens based on a dynamic threshold (scaled by the
        probability of the most likely token.)
                                Defaults to None
        temp (float): Temperature for softmax sampling (temperature sampling (Ackley et al., 1985)):
                        - if >1, increases entropy (randomness)
                        - if <1, decreases entropy (more deterministic)
                        - if 1, untempered distribution

    Returns:
        torch.Tensor: Sampled token ID from the distribution, shape (b, 1)
    """
    assert top_p is None or min_p is None, "Cannot use top_p and min_p together"

    if temp == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)  # keepdim as it is to concat (needs same dim)

    logits = logits / temp  # inplace update to inference tensor outside InferenceMode is not allowed
    probs = torch.softmax(logits, dim=-1)

    # Optional sampling methods
    if min_p:
        min_tokens_to_keep = 1 if top_k is None else top_k
        probs = _min_p_sampling(probs, min_p, min_tokens_to_keep)

    elif top_p:
        probs = _top_p_sampling(probs, top_p, top_k)

    elif top_k:
        probs = _top_k_sampling(probs, top_k)

    probs /= probs.sum(dim=-1, keepdim=True)  # renormalize new distrib/ sum up to 1
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def _top_k_sampling(probs, k):
    """
    Performs top-k sampling on the probabilities by keeping only the k highest probability tokens.
    https://arxiv.org/abs/1805.04833 section 5.4

    Args:
        probs (torch.Tensor): Input distribution tensor representing token probabilities, shape (b, v)
        k (int): Number of top tokens to keep

    Returns:
        torch.Tensor: Sampled token IDs from the distribution, shape (b, 1)
    """
    top_k_probs, top_idx = torch.topk(probs, k)
    # initiate a tensor of the size of probabilities with all values set to 0
    filt_probs = torch.full_like(probs, 0.0)
    # mapping top k values back via their idx, on the given dim (-1), in place
    filt_probs.scatter_(-1, top_idx, top_k_probs)

    return filt_probs


def _top_p_sampling(probs, p, top_k=None):
    """
    Performs top-p nucleus sampling on the probabilities.
    The goal is to find the smallest set of top tokens whose cumulative probability is at least p.
    https://arxiv.org/abs/1904.09751

    Args:
        probs (torch.Tensor): Input distribution tensor representing token probabilities, shape (b, v)
                            or shape (b, draft_max_gen, v) if speculative decoding
        p (float): The cumulative probability threshold for top-p sampling, range [0.0, 1.0].
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.

    Returns:
        torch.Tensor: Sampled token IDs from the distribution, shape (b, 1)
    """
    # limiting top-p to top-k tokens if specified
    if top_k:
        top_k_probs, _ = torch.topk(probs, top_k)

        # correctly slice: for 2D tensor (b, v) normal generation and 3D tensor (b, draft_max_gen, v) spec decoding
        last_top_k_probs = top_k_probs[..., -1].unsqueeze(-1)  # getting last top-k as cutoff point to mask

        top_k_mask = probs < last_top_k_probs
        probs.masked_fill_(top_k_mask, 0.0)

    sorted_probs, og_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)  # CDF

    p_mask = cum_probs > p
    # trick (seen on HF) to keep the pivot/threshold token in the set (by shifting the mask by 1)
    p_mask[..., 1:] = p_mask[..., :-1].clone()
    p_mask[..., 0] = False  # putting back first token in the set (we'll always need the most probable token)
    sorted_probs.masked_fill_(p_mask, 0.0)

    # re-assigning the sorted masked probs to the original indices
    probs.scatter_(-1, og_idx, sorted_probs)

    return probs


def _min_p_sampling(probs, min_p=0.1, min_tokens_to_keep=1):
    """
    Performs min-p sampling on the probabilities.
    The goal is to select tokens dynamically based on a minimum threshold that is proportional to the probability of the
    most likely token (p_max).
    https://arxiv.org/abs/2407.01082

    Note: Not mentioned in the base description, but they use a `min_tokens_to_keep` arg to guarantee at least this
    number of tokens in the distribution, in case the scaled_min_p filters too many tokens.

    args:
        probs (torch.Tensor): Input distribution tensor representing token probabilities, shape (b, v)
                            or shape (b, draft_max_gen, v) if speculative decoding
        min_p (float): base probability threshold (p_base in the paper) range (0,1]. Defaults to 0.1.
        min_tokens_to_keep (int): minimum number of tokens to keep in the distribution, in case the scaled_min_p filters
                                too many tokens. Defaults to 1.
    """
    # get the highest probability for each distrib in the batch
    p_max = torch.amax(probs, dim=-1, keepdim=True)
    # scale the base threshold by p_max
    scaled_min_p = min_p * p_max

    tokens_to_remove = probs < scaled_min_p
    # keep at least `min_tokens_to_keep` tokens regardless of the scaled_min_p
    top_k_idx = torch.topk(probs, min_tokens_to_keep, dim=-1).indices
    tokens_to_remove.scatter_(-1, top_k_idx, False)

    # adjust/truncate the distribs with tokens which have a p >= scaled_min_p
    probs.masked_fill_(tokens_to_remove, 0.0)

    return probs


# test code
if __name__ == "__main__":

    import tiktoken

    import config
    from llm_quest.gpt.gpt_download_weights import download_gpt_model, load_gpt_weights
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import ids_to_text, text_to_ids

    # ---------------------------- PART A ------- Testing simple func generation without pretrained weights

    # print("START")
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)

    # model.eval()
    # model = GPTModel(GPT_CONFIG_124M)

    ## Extracting model parameters for MultiHeadAttention
    # print(sum(p.numel() for p in model.trf_blocks[0].att.parameters()))

    # output = model(batch)
    # print(output.shape)

    # starting_txt = "Hello, i am a model"
    # tokenized_txt = tokenizer.encode(starting_txt)
    ## wrapping in a batch dim as expected input is (b, tok_ids) shape
    # tokenized_tensor = torch.tensor(tokenized_txt).unsqueeze(0)
    # print(tokenized_tensor.shape)

    # output = generate_simple_loop(tokenized_tensor, model, 3, GPT_CONFIG_124M["context_length"])
    # print(output)
    # print(tokenizer.decode(output.squeeze(0).tolist()))

    # ---------------------------- PART B ------- Testing generation with weights trained on "the_verdict"

    # tokenizer = tiktoken.get_encoding("gpt2")

    ## loading our custom pretrained config/model params
    # model = GPTModel(GPT_SMALL_CONFIG)
    # checkpoint = torch.load(config.custom_pretrained_w_gpt2)
    # model.load_state_dict(checkpoint["model_state_dict"])
    #
    # torch.manual_seed(123)
    # model.eval()
    #
    # output2 = generate_loop(
    #    input=text_to_ids("Every effort moves you", tokenizer=tokenizer),
    #    model=model,
    #    max_gen=15,
    #    context_length=GPT_SMALL_CONFIG["context_length"],
    #    top_k=25,
    #    temp=1.4,
    # )
    #
    # print(f"OUTPUT2: {ids_to_text(output2, tokenizer)}")

    # ---------------------------- PART C ------- Testing generation with OpenAI's pretrained weights
    device = config.auto_device

    weights_path = download_gpt_model(gpt_size="gpt_s", save_dir=config.openai_pretrained_w_gpt2_s)

    tokenizer = tiktoken.get_encoding("gpt2")
    model_settings = config.gpt2_config_creator("gpt_s")
    torch.manual_seed(123)

    model = GPTModel(model_settings)
    model.eval()

    load_gpt_weights(model, weights_path)

    model.to(device)  # we move the model to GPU *after* loading weights

    output3 = generate_loop_kv_cache(
        input_tensor=text_to_ids("This is where it", tokenizer=tokenizer).repeat_interleave(3, dim=0),
        model=model,
        max_gen=20,
        context_length=model_settings["context_length"],
        top_k=25,
        top_p=0.95,
        min_p=None,
        temp=1.4,
    )

    for tensor in output3:
        print(ids_to_text(tensor, tokenizer))
