import torch


# NOTE 1: The function version is more readable but slightly less efficient (if `num_grad_updates` > 1) as this forces
# us to:
# - recompute for the old policy Q*log(Q), every gradient step, in the KL(Q || M) term.
#   Ie, KL(Q || M)= ∑Q*log(Q/M) = ∑(Q*log(Q) - Q*log(M))
#   We can't fully precompute KL(Q || M) though, as M, being(P+Q)/2, depends also on the current policy P being updated.
#
# - recompute the new diagonal mask + normalized attention scores: they do not include the token t attending to
#   itself, but [1,... t-1] so we can't use the model attention weights directly, we need to re-apply a mask and
#   renormalize.
#
# NOTE 2: if `num_grad_updates` = 1, there shouldn't be any difference in JSD(P||Q) since the current policy P is
# exactly equal to the old policy Q
#
# NOTE 3: `old_attention_weights` should be the attention weights extracted from the local inference model, not from the
# inference framework (ie, vLLM, ...). Otherwise even at `num_grad_updates` = 1, there will be a difference in JSD(P||Q)
# and gradients will be updated on noisy Inference-Training framwework, which is not what we want.
class AttentionDivergenceLoss(torch.nn.Module):
    """
    Calculates the "Advantage-Weighted attention divergence" loss from the paper:
    Reinforced Attention Learning (RAL) https://arxiv.org/abs/2602.04884

    Normalized attention scores (after softmax) should be retrieved/exposed from the model.
    RAL loss computes the divergence for "generated tokens" only.

    L_ral = Advantage * JSD(P || Q)
    Minimizing L_ral is either done by:
        - pulling the current policy toward the old policy (if advantage > 0, we want to minimize JSD)
        - pushing the current policy away from the old policy (if advantage < 0, we want to maximize JSD).

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
            old_attention_weights: (torch.Tensor), the attention weights from the local old policy Q (not from
                                    a separated inference framework), shape (b, num_heads, seq_len, seq_len)
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
        # we need to clamp because the 1st token, now, is 0.0 and guaranteed to trigger a div by 0
        new_attn_weights = new_attn_weights / new_attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return new_attn_weights.clamp(min=1e-8)

    def forward(self, policy_attention_weights, advantages, loss_mask):
        """
        Calculates the RAL loss for a batch.

        Args:
            policy_attention_weights: (torch.Tensor), shape (b, num_heads, seq_len, seq_len)
            advantages: (torch.Tensor), advantages per sequence, shape (b,)
            loss_mask: (torch.Tensor), can be the same loss mask used for GRPO, here we need it to mask prompt,
                                        shape (b, seq_len)
        """
        if self.q_norm_attn_weights is None or self.qlog_q is None:
            raise RuntimeError("We must call `precompute_qlog_q` before calling `forward`.")

        p_norm_attn_weights = self._prepare_attention_weights(policy_attention_weights, self.diag_mask)

        # M = (P+Q)/2
        m = (p_norm_attn_weights + self.q_norm_attn_weights) / 2.0
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


def attention_divergence_loss(policy_attention_weights, old_attention_weights, advantages, loss_mask, ral_factor=1.0):
    """
    Calculates the "Advantage-Weighted attention divergence" loss from the paper:
    Reinforced Attention Learning (RAL) https://arxiv.org/abs/2602.04884

    Same as `AttentionDivergenceLoss` class but as a 1 call function.

    Args:
        policy_attention_weights: (torch.Tensor), shape (b, num_heads, seq_len, seq_len)
        old_attention_weights: (torch.Tensor), the attention weights from the local old policy Q (not from
                                    a separated inference framework), shape (b, num_heads, seq_len, seq_len)
        advantages: (torch.Tensor), advantages per sequence, shape (b,)
        loss_mask: (torch.Tensor), can be the same loss mask used for GRPO, here we need it to mask prompt,
                                        shape (b, seq_len)
        ral_factor: (float) scaling factor for the RAL loss, recommended values between 0.5 and 1.5

    Returns:
        RAL loss: (torch.Tensor), scalar
    """

    seq_len = policy_attention_weights.shape[-1]
    diag_mask = torch.eye(seq_len, dtype=torch.bool, device=policy_attention_weights.device)

    def _prepare_attn_weights(attn_weights):
        """Average over heads, mask the diagonal, renormalize to sum to 1 and clamp to avoid log(0) later on."""
        attn = attn_weights.mean(dim=1).masked_fill(diag_mask, 0.0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return attn.clamp(min=1e-8)

    with torch.no_grad():
        q_norm_attn_weights = _prepare_attn_weights(old_attention_weights)  # shape (b, seq_len, seq_len)
    p_norm_attn_weights = _prepare_attn_weights(policy_attention_weights)

    # M = (P+Q)/2
    m = (p_norm_attn_weights + q_norm_attn_weights) / 2.0
    log_m = torch.log(m)

    # Use PyTorch builtin kl_div for both KL terms: kl_div(log(M), P) = KL(P || M) with input=log(M), target=P.
    p_kl_div = torch.nn.functional.kl_div(log_m, p_norm_attn_weights, reduction="none")
    q_kl_div = torch.nn.functional.kl_div(log_m, q_norm_attn_weights, reduction="none")

    # Jensen-Shannon divergence: JSD(P || Q) = 0.5 * (KL(P || M) + KL(Q || M))
    # sum over keys/attended tokens to get divergence per query token (b, seq_len, seq_len) → (b, seq_len)
    jsd = 0.5 * (p_kl_div + q_kl_div).sum(dim=-1)

    # ral loss: advantages * JSD
    ral_loss = advantages.unsqueeze(1) * jsd * loss_mask
    ral_loss = ral_loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

    return ral_loss.mean() * ral_factor


# quick inline test
if __name__ == "__main__":
    torch.manual_seed(42)

    b, h, s, _ = 2, 2, 3, 3
    device = "cpu"
    policy_attention_weights = torch.softmax(torch.randn(b, h, s, s, device=device), dim=-1)
    old_attention_weights = torch.softmax(torch.randn(b, h, s, s, device=device), dim=-1)
    advantages = torch.randn(2, device=device)
    loss_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], device=device)

    # function version
    ral_loss_value1 = attention_divergence_loss(
        policy_attention_weights=policy_attention_weights,
        old_attention_weights=old_attention_weights,
        advantages=advantages,
        loss_mask=loss_mask,
        ral_factor=1.0,
    )
    print("ral_loss_value1:", ral_loss_value1)

    # class version
    ral_loss = AttentionDivergenceLoss(ral_factor=1.0)
    ral_loss.precompute_q_and_mask(old_attention_weights)
    ral_loss_value2 = ral_loss(policy_attention_weights, advantages, loss_mask)
    print("ral_loss_value2:", ral_loss_value2)

    assert torch.allclose(ral_loss_value1, ral_loss_value2)
