from llm_quest.alignment.rlhf_grpo.grpo_engine import log_probs_per_token


def log_probs_per_seq(logits, inputs, attention_mask):
    """
    Compute the log probabilities per sequence.

    Args:
        logits (torch.Tensor): Tensor of shape (B*, S*, vocab_size) containing the logits.
        inputs (torch.Tensor): Tensor of shape (B*, S*) containing the generated tokens from the policy.
        attention_mask (torch.Tensor, optional): Tensor of shape (B*, S*) for masking prompt+padding tokens.

        *considering B as batch_size * num_samples and S as prompt_len+max_gen.

    Returns:
        torch.Tensor: Tensor of shape (B,) containing the log probabilities per sequence.
    """
    # here the mask is not optional as we don't want padding+prompt tokens to be considered for the mean
    # (in GRPO, per token, we can delay the masking until the loss calculation)
    tokens_logprobs = log_probs_per_token(logits, inputs, attention_mask)  # shape: (B, S-1)
    seq_logprobs = tokens_logprobs.sum(dim=1) / attention_mask.sum(dim=1)  # shape: (B,)

    return seq_logprobs
