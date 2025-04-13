import torch
import torch.nn as nn

from llm_quest.gpt.gpt_transformer_block import GELU


class Expert(nn.Module):
    """A single expert for the Mixture of Experts (MOE) architecture.

    This class is the same as the GPT2 FFN but allowing modular hidden size with the argument scaling_factor, in order
    to allow more fine-grained control over experts' hidden size.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters. It must include the "emb_dim",
            which specifies the embedding dimension.
        scaling_factor (float): A multiplier used to scale the hidden layer size.
    """

    def __init__(self, cfg, scaling_factor):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], int(4 * cfg["emb_dim"] * scaling_factor)),
            GELU(),
            nn.Linear(int(4 * cfg["emb_dim"] * scaling_factor), cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MoE(nn.Module):
    """Mixture of Experts (MoE) layer.

    This layer implements a sparse MoE, where a gate selects a subset of experts for each token.
    The outputs from the selected experts are then combined to produce the final output.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters. It must include "emb_dim",
            which specifies the embedding dimension.
        num_experts (int): Total number of experts.
        top_k (int): Number of experts to select
        scaling_factor (float or "auto"): Scaling factor for the hidden layer size of each expert:
            - If "auto", will automatically downscale active experts to match GPT2 FFN size
            - If 1, each expert has the same hidden size as the original GPT2 FFN
            - If < 1, each expert has a smaller hidden size than the original GPT2 FFN, inversely for > 1
        load_coeff (float): Coefficient for the load balancing loss.
        z_router_coeff (float): Coefficient for the router z-loss.

    Attributes:
        moe_loss (torch.Tensor): Total moe loss, combining load balancing and router z-loss.
    """

    def __init__(
        self,
        cfg,
        num_experts=8,
        top_k=2,
        scaling_factor="auto",
        load_coeff=10e-2,
        z_router_coeff=1e-3,
    ):
        super().__init__()
        assert (scaling_factor == "auto") or (
            cfg["emb_dim"] % scaling_factor == 0
        ), "emb_dim must be divisible by scaling_factor"
        assert 0 < top_k <= num_experts, "top_k must be > 0 and and <= num_experts"

        if scaling_factor == "auto":
            scaling_factor = 1 / top_k
        self.experts = nn.ModuleList([Expert(cfg, scaling_factor) for _ in range(num_experts)])
        self.gate = nn.Linear(cfg["emb_dim"], num_experts, bias=True)
        self.top_k = top_k
        self.num_experts = num_experts
        self.load_coeff = load_coeff
        self.z_router_coeff = z_router_coeff

    def forward(self, x):
        b, s, emb_dim = x.shape
        # gating
        gate_logits = self.gate(x)  # shape (b,s, num_experts)
        gate_probas = nn.functional.softmax(gate_logits, dim=-1)
        topk_probas, topk_idxs = gate_probas.topk(self.top_k, dim=-1)  # shape (b,s, topk)
        topk_probas /= topk_probas.sum(dim=-1, keepdim=True)  # normalize to topk range

        # z router loss
        z_router_loss = (torch.logsumexp(gate_logits, dim=-1) ** 2).mean()  # avoid overflow with torch.logsumexp()
        # load loss
        counts = torch.bincount(topk_idxs.view(-1), minlength=self.num_experts).to(dtype=x.dtype)
        f_i = counts / (self.top_k * b * s)  # count of tokens dispatched to expert i / number of tokens * topk
        p_i = torch.mean(gate_probas, dim=(0, 1))  # fraction of probas dispatched to expert i / number of tokens
        load_loss = self.num_experts * torch.dot(f_i, p_i)
        # overall loss
        self.moe_loss = self.z_router_coeff * z_router_loss + self.load_coeff * load_loss

        output = torch.zeros_like(x)  # preallocating output size

        # dispatching (unoptimized)
        for expert in range(self.num_experts):
            # computing the mask once for weights and tokens selection
            weight_mask = expert == topk_idxs
            # mask for tokens assigned to the ith expert across all topk
            mask = weight_mask.any(dim=-1)  # shape (b, s, topk) → (b, s)

            if mask.any():
                # retrieving weights for the ith expert across all topk (summed)
                # shape (b, s, topk) → (b, s)
                expert_weights = (topk_probas * weight_mask).sum(dim=-1)

                # select tokens and weights using the same mask
                # shape (b, s, emb_dim) → (num_selected_tokens, emb_dim)
                selected_tokens = x[mask]
                # shape (b, s) → (num_selected_tokens,) → (num_selected_tokens, 1) broadcast for elem-wise mult
                selected_weights = expert_weights[mask].unsqueeze(-1)

                # compute weighted output
                expert_output = self.experts[expert](selected_tokens) * selected_weights
                output[mask] += expert_output

        return output
