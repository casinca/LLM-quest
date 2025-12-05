# logprobs_per_token are computed with log_probs_per_token() in llm_quest.alignment.rlhf_grpo.grpo_engine
# separation of concerns:
# - In case we want to use KL div with GSPO, we keep the logprobs_per_token without having to recompute it later
# - for the policy ratio per sequence, we just need to call this function twice for old/new logprobs.
#
# It would have been more efficient to compute the ratio per token once and then normalize. For flexibility, in case of
# future variants, I keep the normalization per logprobs.
import torch


def log_probs_per_seq(logprobs_per_token, loss_mask):
    """
    Compute the log probabilities per sequence.

    Args:
        logprobs_per_token (torch.Tensor): Tensor of shape (B*, S*) containing the log probabilities per token.
        loss_mask (torch.Tensor, optional): Tensor of shape (B*, S*) for masking prompt+padding tokens, this should be
        the reward mask.

        *considering B as batch_size * num_samples and S as prompt_len+max_gen.

    Returns:
        torch.Tensor: Tensor of shape (B,) containing the log probabilities per sequence.
    """
    # NOTE: here the mask is not optional as we don't want padding+prompt tokens to be considered for the mean
    # (if token-level, we can delay the masking until the loss calculation)
    seq_logprobs = (logprobs_per_token * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)  # shape: (B,)

    return seq_logprobs


def gspo_loss(masked_policy_ratio, advantages, min_clip, max_clip):
    """
    Compute the classic GSPO loss.
    The policy ratio should already be masked for padding+prompt tokens via log_probs_per_seq()

    Args:
        masked_policy_ratio (torch.Tensor): Tensor of shape (B,) containing the policy ratio masked for padding+prompt
        tokens.
        advantages (torch.Tensor): Tensor of shape (B,) containing the advantages.
        min_clip (float): Minimum epsilon clip value.
        max_clip (float): Maximum epsilon clip value.

    Returns:
        torch.Tensor: Tensor of shape (1,) containing the GSPO loss.

    """
    surr_obj = masked_policy_ratio * advantages
    clipped_surr_obj = torch.clip(masked_policy_ratio, min=1 - min_clip, max=1 + max_clip) * advantages

    gspo_loss = -(torch.min(surr_obj, clipped_surr_obj))
    gspo_loss_batch = gspo_loss.mean()

    return gspo_loss_batch
