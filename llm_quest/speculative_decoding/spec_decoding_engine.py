import torch

from llm_quest.gpt.generate import _top_k_sampling, _top_p_sampling, sampling
from llm_quest.gpt.gpt_attention import KVCache


# This is similar to the compute_logprobs() method in the `DPOLoss` class.
def get_logprobs(logits, generated_tokens, temp=0.0):
    """
    Function to calculate and retrieve the log probability of each generated tokens

    args:
        logits (torch.Tensor): The output logits from the model, shape (b, s, v),
        generated_tokens (torch.Tensor): The generated tokens, shape (b, s),
        temp (float): The temperature for the distribution

    Returns:
        torch.Tensor: The log probabilities of the generated tokens, shape (b, s)
    """
    if temp > 0.0:
        logits = logits / temp  # TODO see if safe for inplace and squeeze_

    logprobs = torch.log_softmax(logits, dim=-1)

    gen_tokens_logprobs = torch.gather(
        logprobs,
        dim=-1,
        index=generated_tokens.unsqueeze(-1),  # needs same shape (b, s, 1) as logprobs for gather
    ).squeeze(-1)

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

    if temp > 0.0:
        target_logits = target_logits / temp
        draft_logits = draft_logits / temp

    target_probs = torch.softmax(target_logits, dim=-1)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    zeroes = torch.zeros_like(target_probs)
    adjusted_probs = torch.max(zeroes, target_probs - draft_probs)

    if top_p:
        adjusted_probs = _top_p_sampling(adjusted_probs, top_p, top_k)
    elif top_k:
        adjusted_probs = _top_k_sampling(adjusted_probs, top_k)

    adjusted_probs /= adjusted_probs.sum(dim=-1, keepdim=True)  # renormalize to sum up to 1
    next_token = torch.multinomial(adjusted_probs, num_samples=1)

    return next_token


def speculative_sampling_greedy(target_logits, generated_tokens):
    """
    Simplified route for greedy/deterministic case of speculative sampling, leveraging one hot encoding distributions we
    don't need the draft model's distributions to compute acceptance/rejection.

    args:
    target_logits (torch.Tensor): The logit distributions from the target model for each next-token prediction in the
    draft sequence, shape (b, draft_max_gen+1, v)
    generated_tokens (torch.Tensor): The generated tokens, shape (b, draft_max_gen)

    returns:
        torch.Tensor: The accepted tokens + last token, shape (b, num_accepted + 1)
    """
    accepted_tokens = []
    num_accepted = 0
    num_drafted = generated_tokens.shape[1]

    target_choices = torch.argmax(target_logits[:, :-1, :], dim=-1)

    for i in range(num_drafted):
        if target_choices[0, i] == generated_tokens[0, i]:
            accepted_tokens.append(generated_tokens[0, i].item())
            num_accepted += 1
        else:
            next_token = target_choices[0, i]
            accepted_tokens.append(next_token.item())
            break

    if num_accepted == num_drafted:
        next_token = sampling(target_logits[:, -1, :], temp=0.0)
        accepted_tokens.append(next_token.item())

    accepted_tokens = torch.tensor(accepted_tokens, device=generated_tokens.device).unsqueeze(0)

    return accepted_tokens


# NOTE: Passing distributions (PMF) for acceptance might be weird as we don't need it for comparing sequentially
# single tokens' probabilities p(x_i) and q(x_i).
# However, we still need them in case of:
# - rejection, for the adjusted distribution p'(x) = norm(max(0, p(x) - q(x))).
# - all accepted, for the bonus token sampled from the target model
# And we don't know beforehand which token will be rejected, hence the reason for keeping distributions
# from both models.
def speculative_sampling(
    draft_logits,
    target_logits,
    generated_tokens,
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
        top_k (int | None): limits sampling to top k most likely tokens. None to disable.
        top_p (float | None): limits sampling to top p most likely tokens. Can be combined with top_k. range [0.0, 1.0].
                                None to disable.
        temp (float): The temperature for the distribution, needs to be > 0.0

    returns:
        torch.Tensor: The accepted tokens + last token, shape (b, num_accepted + 1)
    """
    assert temp > 0.0, "Temperature needs to be > 0.0, greedy decoding is handled separately"

    accepted_tokens = []
    num_accepted = 0
    num_drafted = generated_tokens.shape[1]

    draft_logprobs = get_logprobs(draft_logits, generated_tokens, temp)
    target_logprobs = get_logprobs(target_logits[:, :-1, :], generated_tokens, temp)

    ratios = torch.exp(target_logprobs - draft_logprobs)

    for i in range(num_drafted):
        r = torch.rand(1).item()

        # acceptance condition: r < p(x) / q(x), ie if p(x) >= q(x) or probabilistically if p(x) < q(x)
        if r < ratios[0, i]:
            accepted_tokens.append(generated_tokens[0, i].item())
            num_accepted += 1
        else:  # rejection: we sample the last token from the adjusted distribution
            next_token = _rejection_sampling(
                draft_logits[:, i, :],
                target_logits[:, i, :],
                top_k,
                top_p,
                temp,
            )
            accepted_tokens.append(next_token.item())
            break

    if num_accepted == num_drafted:  # all accepted, sample bonus token from target model x ~ p_γ+1(x)
        next_token = sampling(target_logits[:, -1, :], top_k, top_p, temp)
        accepted_tokens.append(next_token.item())

    accepted_tokens = torch.tensor(accepted_tokens, device=generated_tokens.device).unsqueeze(0)

    return accepted_tokens


def speculative_generate(
    target_model,
    draft_model,
    prompt,
    max_gen,
    draft_max_gen,  # this is γ from the paper
    context_length,
    top_k=None,
    top_p=None,
    temp=0.0,
    eos_id=None,
    device="cuda",
):
    """
    # TODO
    """
    # we will use the KVcache for the draft/approximation model to speed it up even more
    # The target model isn't generating tokens, just being passed the prompt+generated tokens for its probabilities
    tokens_ids = []

    kv_cache = KVCache(num_layers=len(draft_model.trf_blocks), max_seq_len=context_length)

    prompt = prompt.to(device)
    prompt_len = prompt.shape[1]
    trunc_prompt = prompt[:, -context_length:]

    with torch.inference_mode():
        draft_logits = draft_model(trunc_prompt, kv_cache=kv_cache)[:, -1, :]

        # --- Drafting tokens with the approximation/draft model ---
        for _ in range(draft_max_gen):
            next_token = sampling(draft_logits, top_k, top_p, temp)

            if eos_id is not None and next_token == eos_id:
                break

            tokens_ids.append(next_token)
            draft_logits = draft_model(next_token, kv_cache=kv_cache).squeeze(1)

        full_tokens = torch.cat([prompt] + tokens_ids, dim=-1)

        # --- Determining the number of accepted tokens ---
        # passing all the tokens (prompt + draft tokens) to the target model to retrieve its probability distributions
        # also passing to the draft model itself (because we used KVcache and had only access to a single token distrib)
        # TODO if refeeding to the draft model is too slow, we'd have to retrieve incrementally during the loop
        # Slicing as we are not interested in prompts logits:
        # - prompt_len - 1 to include the logits distribution of the 1st draft, from the last prompt token
        # - For the target model, we include the prediction from the last token, for the bonus token, shape (b, γ+1, v)
        target_logits = target_model(full_tokens, kv_cache=None)[:, prompt_len - 1 : prompt_len + draft_max_gen :, :]
        generated_tokens = full_tokens[:, prompt_len:]

        if temp == 0.0:
            accepted_tokens = speculative_sampling_greedy(target_logits, generated_tokens)
        else:
            # (inside else block to avoid unnecessary computation if temp == 0.0)
            draft_logits = draft_model(full_tokens, kv_cache=None)[
                :, prompt_len - 1 : prompt_len + draft_max_gen - 1, :
            ]
            accepted_tokens = speculative_sampling(draft_logits, target_logits, generated_tokens, top_k, top_p, temp)

        return torch.cat([prompt, accepted_tokens], dim=-1)
