import torch

from llm_quest.gpt.generate import _top_k_sampling, _top_p_sampling, sampling
from llm_quest.gpt.gpt_attention import KVCache


def get_modified_distrib(logits, top_k, top_p, temp, return_logprobs=False):
    """
    Helper function that takes logits, convert to and return probability distribution based on temperature and filter
    based on top_k/top_p if specified.

    args:
        logits (torch.Tensor): The logits from the model, shape (b, v)
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution
        return_logprobs (bool): Whether to return the log probabilities instead of the probabilities

    returns:
        torch.Tensor: The probabilities or log probabilities of the generated tokens, shape (b, v)
    """
    # logits = logits.float() # check if it makes a difference vs bf16, need to be uniformized everywhere needed
    if temp > 0.0:
        logits = logits / temp

    probs = torch.softmax(logits, dim=-1)

    if top_p:
        probs = _top_p_sampling(probs, top_p, top_k)
    elif top_k:
        probs = _top_k_sampling(probs, top_k)

    probs /= probs.sum(dim=-1, keepdim=True)  # renormalize to sum up to 1

    if return_logprobs:
        return torch.log(probs)

    return probs


# This is similar to the compute_logprobs() method in the `DPOLoss` class.
def get_logprobs(logits, generated_tokens, top_k, top_p, temp=0.0):
    """
    Function to calculate and retrieve the log probability of each generated tokens

    args:
        logits (torch.Tensor): The output logits from the model, shape (b, s, v),
        generated_tokens (torch.Tensor): The generated tokens, shape (b, s),
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution

    Returns:
        torch.Tensor: The log probabilities of the generated tokens, shape (b, s)
    """
    logprobs = get_modified_distrib(logits, top_k, top_p, temp, return_logprobs=True)

    gen_tokens_logprobs = torch.gather(
        logprobs,
        dim=-1,
        index=generated_tokens.unsqueeze(-1),  # needs same shape (b, s, 1) as logprobs for gather
    ).squeeze_(-1)

    return gen_tokens_logprobs


def _rejection_sampling(draft_logits, target_logits, top_k, top_p, temp):
    """
    Function to perform rejection sampling logic if a token is rejected

    args:
        draft_logits (torch.Tensor): The logits distribution from the draft model for the next token, shape (v)
        target_logits (torch.Tensor): The logits distribution from the target model for the next token, shape (v)
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution, needs to be > 0.0

    returns:
        torch.Tensor: The sampled token, shape (1)
    """
    assert temp > 0.0, "Temperature needs to be > 0.0, greedy decoding is handled separately"

    target_probs = get_modified_distrib(target_logits, top_k, top_p, temp)
    draft_probs = get_modified_distrib(draft_logits, top_k, top_p, temp)

    adjusted_probs = torch.clamp_min(target_probs - draft_probs, 0.0)

    adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)  # renormalize to sum up to 1
    next_token = torch.multinomial(adjusted_probs, num_samples=1).squeeze(0)

    return next_token


def speculative_sampling_greedy(target_logits, generated_tokens, remaining_tokens):
    """
    Simplified route for greedy/deterministic case of speculative sampling, leveraging one hot encoding distributions we
    don't need the draft model's distributions to compute acceptance/rejection.

    args:
    target_logits (torch.Tensor): The logit distributions from the target model for each next-token prediction in the
    draft sequence, shape (b, draft_max_gen+1, v)
    generated_tokens (torch.Tensor): The generated tokens, shape (b, draft_max_gen)
    remaining_tokens (int): The number of tokens left to generate

    returns:
        torch.Tensor: The accepted tokens + last token, shape (b, num_accepted + 1)
    """
    # (vectorized overhead wasn't worth for this size and was slower than the loop)
    accepted_tokens = []
    num_accepted = 0
    num_drafted = generated_tokens.shape[1]

    target_choices = torch.argmax(target_logits[:, :-1, :], dim=-1)

    for i in range(num_drafted):
        if target_choices[0, i] == generated_tokens[0, i]:
            accepted_tokens.append(generated_tokens[0, i])
            num_accepted += 1
        else:
            accepted_tokens.append(target_choices[0, i])
            break

    if num_accepted == num_drafted and remaining_tokens > num_drafted:
        bonus_token = torch.argmax(target_logits[:, -1, :], dim=-1)
        accepted_tokens.append(bonus_token.squeeze(0))

    accepted_tokens = torch.stack(accepted_tokens).unsqueeze(0)
    return accepted_tokens


# NOTE: Passing distributions (PMF) for acceptance might be weird as we don't need it for comparing sequentially
# single tokens' probabilities p(x_i) and q(x_i), however, we still need them in case of:
# - rejection, for the adjusted distribution p'(x) = norm(max(0, p(x) - q(x))).
# - all accepted, for the bonus token sampled from the target model
# And we don't know beforehand which token will be rejected, hence the reason for keeping distributions
# from both models.
def speculative_sampling(
    draft_logits,
    target_logits,
    generated_tokens,
    remaining_tokens,
    top_k,
    top_p,
    temp,
):
    """
    Function to perform speculative sampling described in the paper algorithm 1, part 3,4 and 5:
    - Determine the number of accepted guesses n.
    - Adjust the distribution from M^p if needed
    - Return one token from Mp, and n tokens from Mq

    args:
        draft_logits (torch.Tensor): The logit distributions from the draft model for each next-token prediction in the
        draft sequence, shape (b, draft_max_gen, v)
        target_logits (torch.Tensor): The logit distributions from the target model for each next-token prediction in the
        draft sequence, shape (b, draft_max_gen+1, v)
        generated_tokens (torch.Tensor): The generated tokens, shape (b, draft_max_gen)
        remaining_tokens (int): The number of tokens left to generate
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution, needs to be > 0.0

    returns:
        torch.Tensor: The accepted tokens + last token, shape (b, num_accepted + 1)
    """
    assert temp > 0.0, "Temperature needs to be > 0.0, greedy decoding is handled separately"

    num_accepted = 0
    accepted_tokens = []
    num_drafted = generated_tokens.shape[1]
    random_values = torch.rand(num_drafted, device=generated_tokens.device)

    draft_logprobs = get_logprobs(draft_logits, generated_tokens, top_k, top_p, temp)
    target_logprobs = get_logprobs(target_logits[:, :-1, :], generated_tokens, top_k, top_p, temp)

    ratios = torch.exp(target_logprobs - draft_logprobs)

    for i in range(num_drafted):
        r = random_values[i]

        # acceptance condition: r < p(x) / q(x), ie if p(x) >= q(x) or probabilistically if p(x) < q(x)
        if r < ratios[0, i]:
            accepted_tokens.append(generated_tokens[0, i].squeeze())
            num_accepted += 1
        else:  # rejection: we sample the last token from the adjusted distribution
            next_token = _rejection_sampling(
                draft_logits[:, i, :],
                target_logits[:, i, :],
                top_k,
                top_p,
                temp,
            )
            accepted_tokens.append(next_token.squeeze())
            break

    # all accepted, sample bonus token from target model x ~ p_γ+1(x)
    if num_accepted == num_drafted and remaining_tokens > num_drafted:
        bonus_token = sampling(target_logits[:, -1, :], top_k, top_p, temp)
        accepted_tokens.append(bonus_token.squeeze())

    accepted_tokens = torch.stack(accepted_tokens).unsqueeze(0)
    return accepted_tokens


def _speculative_step(
    target_model,
    draft_model,
    current_sequence,
    draft_max_gen,
    remaining_tokens,
    context_length,
    top_k=None,
    top_p=None,
    temp=0.0,
    eos_id=None,
):
    """
    Speculative orchestrator: Performs a single step of speculative decoding.

    This function uses the draft model to generate `draft_max_gen` candidate tokens, we then verify these tokens with
    the target model via the `speculative_sampling` function to determine which drafted tokens are accepted.

    Args:
        target_model: The target (larger) model for verification
        draft_model: The draft/approximation (smaller) model for speculation
        current_sequence (torch.Tensor): Input sequence tokens, shape (b, prompt_len + previously generated tokens)
        draft_max_gen (int): Number of tokens to draft per iteration (γ in the paper)
        remaining_tokens (int): The number of tokens left to generate
        context_length (int): Maximum context length for the draft model
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution, needs to be > 0.0
        eos_id (int): End of sequence token ID

    Returns:
        accepted_tokens(torch.Tensor): tensor containing the tokens accepted in this speculative step, shape (b,
        accepted_tokens)
    """
    draft_tokens = []
    draft_logits = []
    curr_len = current_sequence.shape[1]

    kv_cache = KVCache(num_layers=len(draft_model.trf_blocks), max_seq_len=context_length)
    trunc_seq = current_sequence[:, -context_length:] if curr_len > context_length else current_sequence

    drafted_logits = draft_model(trunc_seq, kv_cache=kv_cache)[:, -1, :]  # fill the cache
    draft_logits.append(drafted_logits.unsqueeze(1))

    # --- Drafting tokens with the approximation/draft model ---
    for _ in range(draft_max_gen):
        next_token = sampling(drafted_logits, top_k, top_p, temp)

        if eos_id is not None and next_token.item() == eos_id:
            draft_tokens.append(next_token)
            break

        draft_tokens.append(next_token)
        drafted_logits = draft_model(next_token, kv_cache=kv_cache)[:, -1, :]
        draft_logits.append(drafted_logits.unsqueeze(1))

    full_sequence = torch.cat([current_sequence] + draft_tokens, dim=-1)

    # --- Determining the number of accepted tokens ---
    # slicing to retrieve only the logits distributions for the drafted tokens:
    # - curr_len - 1 to include the logits distribution of the 1st draft, from the
    # - For the target model, we include the prediction from the last token, for the bonus token,
    drafted_sequence = full_sequence[:, curr_len:]
    drafted_len = drafted_sequence.shape[1]

    target_logits = target_model(full_sequence, kv_cache=None)
    target_logits = target_logits[:, curr_len - 1 : curr_len + drafted_len :, :]  # shape (b, γ+1, v)

    if temp == 0.0:
        accepted_tokens = speculative_sampling_greedy(target_logits, drafted_sequence, remaining_tokens)
    else:
        draft_logits_tensor = torch.cat(draft_logits[:drafted_len], dim=1)  # shape (b, γ, v)
        # (get draft logits inside else block to avoid unnecessary computation if temp == 0.0)
        # draft_logits_tensor = draft_logits_tensor[:, curr_len - 1 : curr_len + drafted_len - 1, :]
        accepted_tokens = speculative_sampling(
            draft_logits_tensor, target_logits, drafted_sequence, remaining_tokens, top_k, top_p, temp
        )

    return accepted_tokens


def speculative_generate(
    target_model,
    draft_model,
    prompt,
    max_gen,
    draft_max_gen,
    context_length,
    top_k=None,
    top_p=None,
    temp=0.0,
    eos_id=None,
    device="cuda",
):
    """
    Full speculative decoding generation loop.

    Args:
        target_model: The target (larger) model for verification
        draft_model: The draft/approximation (smaller) model for speculation
        prompt (torch.Tensor): Input prompt tokens, shape (1, prompt_len)
        max_gen (int): Maximum number of tokens to generate
        draft_max_gen (int): Number of tokens to draft per iteration (γ in the paper)
        context_length (int): Maximum context length for the draft model
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution, needs to be > 0.0
        eos_id (int): End of sequence token ID
        device (str): Device to run on

    Returns:
        torch.Tensor: Generated sequence including prompt, shape (1, prompt_len + generated length/max_gen)
    """
    current_sequence = prompt.to(device)
    tokens_gen = 0

    with torch.inference_mode():

        # --- Main generation loop ---
        while tokens_gen < max_gen:
            # calc tokens left to generate
            remaining_tokens = max_gen - tokens_gen
            curr_draft_max = min(draft_max_gen, remaining_tokens)  #  case remaining_tokens < draft_max_gen

            if curr_draft_max <= 0:
                break

            # one iteration of speculative decoding
            new_tokens = _speculative_step(  # shape (b, <= curr_draft_max+1) "<" if EoS is hit
                target_model=target_model,
                draft_model=draft_model,
                current_sequence=current_sequence,
                draft_max_gen=curr_draft_max,
                remaining_tokens=remaining_tokens,
                context_length=context_length,
                top_k=top_k,
                top_p=top_p,
                temp=temp,
                eos_id=eos_id,
            )

            # update sequence and gen counter
            current_sequence = torch.cat([current_sequence, new_tokens], dim=1)
            tokens_gen += new_tokens.shape[1]

            # check for EoS token
            if eos_id is not None and new_tokens.shape[1] > 0 and new_tokens[0, -1].item() == eos_id:
                break

        return current_sequence
