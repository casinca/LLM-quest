import torch
import torch.nn as nn

# Compared to classic_moe.py, we removed the scaling_factor argument and hardcoded the MoE hidden size directly from the
# Qwen3 config.
# We also remove the Z router loss, to keep only the (LBL) auxiliary loss, actually it's the "global-batch" LBL variant
# that Qwen use, but no distributed training here, so Global LBL reduces to the classic LBL already implemented.
#
# Added optional weighted shared expert in Qwen3MoE to make it compatible with Qwen3-Next MoE
# This is actually part c) of my experimental "weighting shared experts" with a single expert:
# https://github.com/casinca/LLM-quest/blob/master/llm_quest/experimental/weighting_shared_experts/Readme.md
#
# Added sigma-MoE initialization for the router/gate weights for Qwen3-Next MoE


def router_weights_init(weights):
    """
    re-initialize router/gate weights following sigma-MoE initialization used in Qwen3-Next
    https://arxiv.org/abs/2310.10837

    The goal is to have a better starting point for training where initial routing of experts is based on cosine
    similarity of the dot product only (scaled by input ||x||) and not impacted by the magnitude of initialized
    weights.

    Args:
        weights (torch.Tensor): router/gate weights
    """
    with torch.no_grad():
        og_std = weights.std()  # saving original SD to scale back later

        l2_norms = torch.linalg.vector_norm(weights, dim=-1, ord=2, keepdim=True)
        weights *= l2_norms.reciprocal()

        weights *= og_std / weights.std()  # rescale to original SD


class Expert(nn.Module):
    """A single expert for the Mixture of Experts (MOE) architecture.

    A modular gated FFN with custom activation function.
    This class is the same as Llama FFN or other popular gated FFN in dense models.
    Here the MoE hidden size is hardcoded directly from the config and not via my scaling_factor argument.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters. It must include the "emb_dim",
            which specifies the embedding dimension.
        hidden_dim (int): Hidden dimension of the expert. If None, will use the moe_hidden_dim from the config.
        activation (torch.nn.functional): Built-in activation function from Pytorch.
    """

    def __init__(self, cfg, hidden_dim=None, activation=torch.nn.functional.silu):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else cfg["moe_hidden_dim"]

        self.lin1 = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.activation = activation
        self.lin_gate = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.lin2 = nn.Linear(self.hidden_dim, cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.activation(x_gate)
        x1 = self.lin1(x)

        return self.lin2(x1 * x_gate)


class Qwen3MoE(nn.Module):
    """Mixture of Experts (MoE) layer.

    This layer implements a sparse MoE, where a gate selects a subset of experts for each token.
    The outputs from the selected experts are then combined to produce the final output.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters, it must include:
        "emb_dim", "num_experts", "top_k", "aux_loss_coef".
    Attributes:
        moe_loss (torch.Tensor): Total moe loss, combining load balancing and router z-loss.
    """

    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg["top_k"]
        self.num_experts = cfg["num_experts"]
        self.load_coeff = cfg["aux_loss_coef"]
        self.training = cfg["training"]

        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg["num_experts"])])
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # For Qwen3-Next MoE: weighted shared expert + sigma-MoE init of router weights
        self.shared_expert_hidden_dim = cfg.get("shared_expert_hidden_dim", None)
        if self.shared_expert_hidden_dim is not None:
            self.shared_expert = Expert(cfg, hidden_dim=self.shared_expert_hidden_dim)
            self.shared_expert_gate = nn.Linear(cfg["emb_dim"], 1, bias=False, dtype=cfg["dtype"])  # single scalar

            # only relevant for Pretraining with Qwen3-Next.
            if self.training:
                print("# WARNING: Training flag, router weights are being re-initialized, shouldn't be used for SFT! ")
                router_weights_init(self.gate.weight)

    def forward(self, x):
        b, s, emb_dim = x.shape
        # gating
        x_2d = x.view(-1, emb_dim)  # matrix/tabular view for easier indexing row/col manipulation (HF style)
        gate_logits = self.gate(x_2d)  # shape (b*s, num_experts)
        gate_probas = nn.functional.softmax(gate_logits, dim=-1)
        topk_probas, topk_idxs = gate_probas.topk(self.top_k, dim=-1)  # shape (b*s, topk)
        topk_probas /= topk_probas.sum(dim=-1, keepdim=True)  # normalize to topk range

        # load loss
        if self.training:
            counts = torch.bincount(topk_idxs.view(-1), minlength=self.num_experts).to(dtype=x.dtype)
            f_i = counts / (self.top_k * b * s)  # count of tokens dispatched to expert i / number of tokens * topk
            p_i = torch.mean(gate_probas, dim=0)  # fraction of probas dispatched to expert i / number of tokens
            load_loss = self.num_experts * torch.dot(f_i, p_i)
            self.moe_loss = self.load_coeff * load_loss

        output = torch.zeros_like(x_2d)  # preallocating output size

        # Optimized implementation: We only loop through activated/hit experts and use atomic writes via `index_add_`
        # create a mask of one hot `num_experts` matrices where True/1 position means:
        # which top-k slot (col) and token (row) are assigned to that expert
        # shape (b*s, topk, num_experts) → (num_experts, topk, b*s)
        expert_mask = torch.nn.functional.one_hot(topk_idxs, num_classes=self.num_experts).permute(2, 1, 0)
        # find which experts (via their idx) are actually used
        expert_hit_count = expert_mask.sum(dim=(-1, -2))
        expert_hit_idx = torch.where(expert_hit_count > 0)[0]  # [0] to unpack single elem tuple

        # dispatching
        for idx in expert_hit_idx:
            # retrieve the current expert's one hot mask
            expert_assignment = expert_mask[idx]  # shape (topk, b*s)
            # retrieve token's coordinates assigned to the current expert
            topk_pos, token_idx = torch.where(expert_assignment)

            # retrieve selected tokens and weights via the above indices
            selected_tokens = x_2d[token_idx]  # shape (num_selected_tokens, emb_dim)
            selected_weights = topk_probas[token_idx, topk_pos].unsqueeze(-1)  # shape (num_selected_tokens, 1)

            # compute weighted expert res and update via index, efficiently, the preallocated output tensor
            expert_output = self.experts[idx](selected_tokens) * selected_weights
            output.index_add_(dim=0, index=token_idx, source=expert_output)

        # optionally, adding weighted shared expert output
        if self.shared_expert_hidden_dim is not None:
            shared_expert_output = self.shared_expert(x_2d)
            scaled_shared_expert_weights = nn.functional.sigmoid(self.shared_expert_gate(x_2d))
            weighted_shared_expert_output = shared_expert_output * scaled_shared_expert_weights

            output += weighted_shared_expert_output

        output = output.view(b, s, emb_dim)  # reshape back to original 3D shape
        return output
