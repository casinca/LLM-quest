import torch

from llm_quest.gpt.generate import sampling
from llm_quest.gpt.gpt_attention import KVCache


def speculative_sampling():
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
    trunc_prompt = prompt[:, -context_length:]

    with torch.inference_mode():
        draft_logits = draft_model(trunc_prompt, kv_cache=kv_cache)[:, -1, :]

        for _ in range(draft_max_gen):
            next_token = sampling(draft_logits, top_k, top_p, temp)

            if eos_id is not None and next_token == eos_id:
                break

            tokens_ids.append(next_token)
            draft_logits = draft_model(next_token, kv_cache=kv_cache).squeeze(1)

        full_tokens = torch.cat([prompt] + tokens_ids, dim=-1)

        # passing all the tokens (prompt + draft tokens) to the target model to retrieve its probability distributions
        # also passing to the draft model itself (because we used KVcache and had only access to a single token distrib)
        target_logits = target_model(full_tokens, kv_cache=None)
        draft_logits = draft_model(full_tokens, kv_cache=None)
