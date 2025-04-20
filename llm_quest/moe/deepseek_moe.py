import torch
import torch.nn as nn

from llm_quest.gpt.gpt_transformer_block import GELU
from llm_quest.gpt_to_llama3.llama_transformer_block import SiLU


class Expert(nn.Module):
    """A single expert for the Mixture of Experts (MOE) architecture.

    This class is the same as Llama FFN but allowing modular hidden size with the argument scaling_factor, in order to
    allow more fine-grained control over experts.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters. It must include the "emb_dim",
            which specifies the embedding dimension.
        scaling_factor (float): A multiplier used to scale the hidden layer size.
    """

    def __init__(self, cfg, scaling_factor):
        super().__init__()
        self.lin1 = nn.Linear(cfg["emb_dim"], int(scaling_factor * cfg["hidden_dim"]), dtype=cfg["dtype"], bias=False)
        self.silu_activ = SiLU()
        self.lin_gate = nn.Linear(
            cfg["emb_dim"], int(scaling_factor * cfg["hidden_dim"]), dtype=cfg["dtype"], bias=False
        )
        self.lin2 = nn.Linear(int(scaling_factor * cfg["hidden_dim"]), cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = self.silu_activ(x1)
        x2 = self.lin_gate(x)

        return self.lin2(x1 * x2)


class Expert_GeLU(nn.Module):
    """A single expert for the Mixture of Experts (MOE) architecture.

    This class is the same as GPT2 FFN but allowing modular hidden size with the argument scaling_factor, in order to
    allow more fine-grained control over experts.

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


class DeepSeekMoE(nn.Module):

    def __init__(
        self,
        cfg,
        num_experts=8,
        num_shared_experts=1,
        top_k=3,
        scaling_factor="auto",
        bias_update_rate=1e-3,
    ):
        super().__init__()
        # some basic checks
        assert (
            scaling_factor == "auto" or cfg["emb_dim"] % scaling_factor == 0
        ), "emb_dim must be divisible by scaling_factor"
        assert (
            top_k != 1 and (num_shared_experts + top_k) % 2 == 0
        ), "the total num of 'active' experts, shared+routed(topk), should be even"
        assert 0 < top_k <= num_experts - num_shared_experts, "top_k must be > 0 and <= routed experts"

        if scaling_factor == "auto":
            scaling_factor = 1 / (top_k + num_shared_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared = num_shared_experts
        self.num_routed = num_experts - num_shared_experts

        # separate experts into shared and routed groups
        self.routed_experts = nn.ModuleList([Expert(cfg, scaling_factor) for _ in range(self.num_routed)])
        self.shared_experts = nn.ModuleList([Expert(cfg, scaling_factor) for _ in range(self.num_shared)])

        # init gate and biases for routed experts only
        self.gate = nn.Linear(cfg["emb_dim"], self.num_routed, bias=True)
        self.register_buffer("biases", torch.zeros(self.num_routed))
        self.bias_update_rate = bias_update_rate

    def forward(self, x):
        # b, s, emb_dim = x.shape
        output = torch.zeros_like(x)  # preallocating output

        # process shared experts (always active)
        for expert in self.shared_experts:
            output += expert(x)

        # gating
        gate_logits = self.gate(x)  # shape (b,s, num_routed)
        gate_probas = nn.functional.softmax(gate_logits, dim=-1)  # we want unbiased probas for weighting
        # adding biases for load balance and top-k experts selection
        biased_probas = gate_probas + self.biases
        topk_idxs = biased_probas.topk(self.top_k, dim=-1)[1]  # shape (b,s, topk)
        # retrieving topk probas and normalizing to topk range
        topk_probas = torch.gather(gate_probas, dim=-1, index=topk_idxs)
        topk_probas /= topk_probas.sum(dim=-1, keepdim=True)

        # dispatching (unoptimized)
        for expert_idx in range(self.num_routed):
            # computing the mask once for weights and tokens selection
            weight_mask = expert_idx == topk_idxs
            # mask for tokens assigned to the ith expert across all topk
            mask = weight_mask.any(dim=-1)  # shape (b, s, topk) → (b, s)

            if mask.any():
                # retrieving weights for the ith expert across all topk (summed)
                # shape (b, s, topk) → (b, s)
                expert_weights = (topk_probas * weight_mask).sum(dim=-1)

                # select tokens and weights using the same mask
                # shape (b, s, emb_dim) → (num_selected_tokens, emb_dim)
                selected_tokens = x[mask]
                # shape (b, s) → (num_selected_tokens) → (num_selected_tokens, 1) broadcast for elem-wise mult
                selected_weights = expert_weights[mask].unsqueeze(-1)

                # compute weighted output
                expert_output = self.routed_experts[expert_idx](selected_tokens) * selected_weights
                output[mask] += expert_output

        # updating biases with the violation error per the paper
        # counts is a vector of the number of tokens dispatched to each expert in the batch, shape (num_routed)
        counts = torch.bincount(topk_idxs.view(-1), minlength=self.num_routed).float()  # float for mean()
        vio_error = counts.mean() - counts
        self.biases += self.bias_update_rate * vio_error.sign()
        self.max_vio = max_violation_batch(counts)  # optional to calc DeepSeek's max violation metric

        return output


# Optional metric made by DeepSeek to measure load balance with their violation error metric
def max_violation_batch(load):
    """
    calc the max violation for a batch:
    max-violation = (max(load) - mean(load)) / mean(load)

    Args:
        load (torch.Tensor): the number of tokens dispatched to each expert

    """
    mean_vio = load.mean()
    max_vio = (load.max() - mean_vio) / mean_vio
    return max_vio


def max_violation_step(model=None):
    """
    calc the max violation for a single step
    """

    max_vios, num_layers = 0.0, 0
    for module in model.modules():
        ffn = getattr(module, "ffn", None)
        if ffn is not None and hasattr(ffn, max_vios):
            max_vios += module.moe.self.max_vio
            num_layers += 1

    return max_vios / num_layers
