import math

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
        self.hidden_dim = int(scaling_factor * cfg["hidden_dim"])

        self.lin1 = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.silu_activ = SiLU()
        self.lin_gate = nn.Linear(cfg["emb_dim"], self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.lin2 = nn.Linear(self.hidden_dim, cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

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
        self.hidden_dim = int(4 * scaling_factor * cfg["emb_dim"])

        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], self.hidden_dim),
            GELU(),
            nn.Linear(self.hidden_dim, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class VectorizedLinear(nn.Module):
    """
    This class creates a batch of `num_experts` linear layers and biases, as if we had multiple nn.Linear in a batch
    dimension.
    """

    def __init__(self, num_experts, in_features, out_features, bias=True):
        """
        Args:
            num_experts (int): Number of experts (batch dimension).
            in_features (int): Number of input features .
            out_features (int): Number of output features.
            bias (bool): Whether to use bias.
        """
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty(num_experts, in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.bias = None

        self._kaiming_init()

    def _kaiming_init(self):
        """
        Kaiming uniform initialization:
        PyTorch copy pasta for initializing the weights of the linear layers and biases, the same way as nn.Linear:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L117-L128
        """
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        assert x.ndim == 4, "x should be (b, num_experts, s, in_features)"  # broadcasted if num_experts is 1
        # ex shapes with first linear `self.lin1`:
        # x.shape: (b, 1, s, emb_dim) and weight.shape: (num_experts, emb_dim, hidden_dim/out_features)
        # output.shape we want (b, num_experts, s, hidden_dim)
        # alt with einsum: x = torch.einsum("bnse,neh->bnsh", x, self.weight)
        x = x @ self.weight

        if self.bias is not None:
            # bias shape: (num_experts, hidden_dim) needs to be broadcasted to (1, num_experts, 1, hidden dim)
            x += self.bias.unsqueeze(0).unsqueeze(2)

        return x  # shape (b, num_experts, s, hidden_dim)


class VectorizedSharedExperts(nn.Module):
    """
    This class is the same as having a nn.ModuleList of `num_experts` Expert modules.
    Since shared experts have the same size and are always active, we can vectorize the computation, instead of doing a
    for loop previously:
            for expert in self.shared_experts:
                output += expert(x)

    Now we do a batched `num_experts` feedforward pass.
    """

    def __init__(self, num_experts, cfg, scaling_factor, bias=True):
        """
        Args:
            num_experts (int): Number of experts (batch dimension).
            cfg (dict): Config dictionary containing model hyperparameters. It must include the "emb_dim",
                which specifies the embedding dimension.
            scaling_factor (float): A multiplier used to scale the hidden layer size.
            bias (bool): Whether to use bias.
        """
        super().__init__()
        self.hidden_dim = int(scaling_factor * cfg["hidden_dim"])

        # we don't use premade nn.Linear layers anymore, we have to build the linear layers and bias manually to get a
        # batch of "num_experts" linear layers and biases, done with `VectorizedLinear`
        self.lin1 = VectorizedLinear(num_experts, cfg["emb_dim"], self.hidden_dim, bias)
        self.silu = nn.functional.silu  # using pytorch's built-in SiLU than my own for speed
        self.lin2 = VectorizedLinear(num_experts, self.hidden_dim, cfg["emb_dim"], bias)

    def forward(self, x):
        assert x.ndim == 3, "x should be (b, s, emb_dim)"

        x = x.unsqueeze(1)  # unsqueeze to add the num_experts dimension (b, 1, s, emb_dim)

        x = self.lin1(x)
        x = self.silu(x)
        x = self.lin2(x)

        return torch.sum(x, dim=1)  # sum over the num_experts dimension to get back (b, s, emb_dim)


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
        self.shared_experts = VectorizedSharedExperts(self.num_shared, cfg, scaling_factor)

        # init gate and biases for routed experts only
        self.gate = nn.Linear(cfg["emb_dim"], self.num_routed, bias=True)
        self.register_buffer("biases", torch.zeros(self.num_routed))
        self.bias_update_rate = bias_update_rate

    def forward(self, x):
        # b, s, emb_dim = x.shape
        output = torch.zeros_like(x)  # preallocating output

        # process shared experts (always active)
        output += self.shared_experts(x)

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
