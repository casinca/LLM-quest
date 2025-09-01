import torch

from llm_quest.gpt.generate import sampling
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


# NOTE: Passing full (PMF) distributions for acceptance might be weird as we don't need it for comparing sequentially
# single tokens' probabilities p(x_i) and q(x_i). However, we still need them, in case of rejection, for the adjusted
# distribution p'(x) = norm(max(0, p(x) - q(x))).
# That's because we don't know beforehand which token will be rejected, hence the reason for keeping full distributions
# from both models.
def speculative_sampling(draft_logits, target_logits, generated_tokens, temp=0.0):
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
        temp (float): The temperature for the distribution

    returns: TODO
    """
    draft_logprobs = get_logprobs(draft_logits, generated_tokens, temp)
    target_logprobs = get_logprobs(target_logits[:, :-1, :], generated_tokens, temp)

    ratios = torch.exp(target_logprobs - draft_logprobs)

    for i in range(ratios.shape[1]):
        pass


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

    kv_cache = KVCache(
        num_layers=len(draft_model.trf_blocks),
        max_seq_len=context_length,
    )

    prompt = prompt.to(device)
    prompt_len = prompt.shape[1]
    trunc_prompt = prompt[:, -context_length:]

    # --- Drafting tokens with the approximation/draft model ---
    with torch.inference_mode():
        draft_logits = draft_model(trunc_prompt, kv_cache=kv_cache)[:, -1, :]

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
        draft_logits = draft_model(full_tokens, kv_cache=None)[:, prompt_len - 1 : prompt_len + draft_max_gen - 1, :]
        target_logits = target_model(full_tokens, kv_cache=None)[:, prompt_len - 1 : prompt_len + draft_max_gen :, :]
        generated_tokens = full_tokens[:, prompt_len:]

        speculative_sampling(draft_logits, target_logits, generated_tokens, temp)
