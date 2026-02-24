import torch


# NOTE: We could have used a single function to compute the loss but it is slightly inefficient as this would force us
# to :
# - recompute for the old policy Q*log(Q), every gradient step, in the KL(Q || M) term.
#   Ie, KL(Q || M)= ∑Q*log(Q/M) = ∑(Q*log(Q) - Q*log(M))
#   We can't fully precompute KL(Q || M) though, as M, being(P+Q)/2, depends also on the current policy P being updated.
#
# - recompute the new diagonal mask + normalized attention scores: they do not include the token t attending to
#   itself, but [1,... t-1] so we can't use the model attention weights directly, we need to re-apply a mask and
#   renormalize.
class AttentionDivergenceLoss(torch.nn.Module):
    """
    Calculates the "Advantage-Weighted attention divergence" loss from the paper:
    Reinforced Attention Learning (RAL) https://arxiv.org/abs/2602.04884

    Normalized attention scores (after softmax) should be retrieved/exposed from the model.
    RAL loss computes the divergence for "generated tokens" only.

    Args:
        ral_factor: (float) scaling factor for the RAL loss, recommended values between 0.5 and 1.5

    Returns:
        RAL loss: (torch.Tensor), scalar
    """

    def __init__(self, ral_factor=1.0):
        super().__init__()
        self.ral_factor = ral_factor

        self.diag_mask = None
        self.q_norm_attn_weights = None
        self.qlog_q = None

    @torch.no_grad()
    def precompute_q_and_mask(self, old_attention_weights):
        """
        This methods serves to:
        - precompute the diagonal mask for the class (also used for the new policy)
        - compute the new normalized attention weights for the old policy Q
        - precompute the first term of the KL(Q || M) divergence: Q*log(Q) to avoid recomputing it every gradient step

        Attention weights are already masked from the model, but not for tokens attending to themselves, so we need to
        create and apply a new mask for the diagonal (token t attending to itself)

        Args:
            old_attention_weights: (torch.Tensor), shape (b, num_heads, seq_len, seq_len)
        """
        seq_len = old_attention_weights.shape[-1]

        self.diag_mask = torch.eye(seq_len, dtype=torch.bool, device=old_attention_weights.device)
        self.q_norm_attn_weights = self._prepare_attention_weights(old_attention_weights, self.diag_mask)
        self.qlog_q = self.q_norm_attn_weights * torch.log(self.q_norm_attn_weights)

    @staticmethod
    def _prepare_attention_weights(attention_weights, diag_mask):
        """
        Average over heads, mask the diagonal, renormalize to sum to 1 and clamp to avoid log(0) later on.

        Args:
            attention_weights: (torch.Tensor), shape (b, num_heads, seq_len, seq_len)
            diag_mask: (torch.Tensor), shape (seq_len, seq_len) to mask token t attending to itself

        Returns:
            new normalized attention weights: (torch.Tensor), shape (b, seq_len, seq_len)
        """
        new_attn_weights = attention_weights.mean(dim=1).masked_fill(diag_mask, 0.0)
        # we need to clamp because the 1st token now is 0.0 and guaranteed to trigger a div by 0
        new_attn_weights = new_attn_weights / new_attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return new_attn_weights.clamp(min=1e-8)

    def forward(self, policy_attention_weights, advantages, loss_mask):
        """
        Calculates the RAL loss for a batch.

        Args:
            policy_attention_weights: (torch.Tensor), shape (b, num_heads, seq_len, seq_len)
            advantages: (torch.Tensor), advantages per sequence, shape (b,)
            loss_mask: (torch.Tensor), shape (b, seq_len) to mask prompt primarily
        """
        if self.q_norm_attn_weights is None or self.qlog_q is None:
            raise RuntimeError("We must call `precompute_qlog_q` before calling `forward`.")

        p_norm_attn_weights = self._prepare_attention_weights(policy_attention_weights, self.diag_mask)

        # M = (P+Q)/2
        m = ((p_norm_attn_weights + self.q_norm_attn_weights) / 2.0).clamp(min=1e-8)
        log_m = torch.log(m)

        # KL(Q || M) = ∑Q*log(Q/M) = ∑(Q*log(Q) - Q*log(M))
        q_kl_div = self.qlog_q - self.q_norm_attn_weights * log_m
        # KL(P || M) = ∑P*log(P/M)
        p_kl_div = p_norm_attn_weights * (torch.log(p_norm_attn_weights) - log_m)
        # alt using pytorch builtin
        # p_kl_div = torch.nn.functional.kl_div(log_m, p_norm_attn_weights, reduction="none")

        # JSD(P || Q) = 0.5 * (KL(P || M) + KL(Q || M))
        # sum over keys/attended tokens to get divergence per query token (b, seq_len, seq_len) → (b, seq_len)
        jsd = 0.5 * (p_kl_div + q_kl_div).sum(dim=-1)

        # ral loss: advantages * JSD
        ral_loss = advantages.unsqueeze(1) * jsd * loss_mask
        ral_loss = ral_loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

        return ral_loss.mean() * self.ral_factor
