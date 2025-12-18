import torch
import torch.nn as nn


class SquaredReLU(nn.Module):
    """
    A ReLU activation but squared.
    paper: https://arxiv.org/abs/2109.08668

    Nvidia Nemotron 3 uses this activation function in the FFN.
    """

    def forward(self, x):
        return torch.square(torch.relu(x))


class Expert(nn.Module):
    """
    Same class as in qwen3_moe.py, but with the activation function changed to SquaredReLU and adding input_dim argument
    for the Linear layers.

    Args:
        cfg (dict): Config dictionary containing model hyperparameters.
        input_dim (int): Input dimension of the expert. If None, will use the emb_dim from the config.
        hidden_dim (int): Hidden dimension of the expert. If None, will use the moe_hidden_dim from the config.
        activation (torch.nn.functional): activation function to use
    """

    def __init__(self, cfg, input_dim=None, hidden_dim=None, activation=SquaredReLU):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else cfg["moe_hidden_dim"]
        self.input_dim = input_dim if input_dim is not None else cfg["emb_dim"]

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.activation = activation()
        self.lin_gate = nn.Linear(self.input_dim, self.hidden_dim, dtype=cfg["dtype"], bias=False)
        self.lin2 = nn.Linear(self.hidden_dim, self.input_dim, dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.activation(x_gate)
        x1 = self.lin1(x)

        return self.lin2(x1 * x_gate)


class LatentMoE(nn.Module):
    """
    Nvidia LatentMoE re-implementation based solely on the Nvidia Nemotron 3 paper:
    https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf

    This is a slightly modified `deepseek_moe.py` with a single shared expert, added latent projection layers and a
    latent factor to control the scaling of experts.
    """

    def __init__(
        self,
        cfg,
        top_k=2,
        num_experts=4,
        routed_expert_hidden_dim=None,
        shared_expert_hidden_dim=None,
        routed_scaling_factor=2.5,  # increase signal/scale up for gradient (unrelated to LatentMoE)
        latent_ratio=4,  # emb_dim / latent_dim (default to 4 per Nvidia Nemotron 3)
        bias_update_rate=1e-3,  # aux-loss free load balancing rate
    ):
        super().__init__()
        # we scale up by latent_ratio the experts, in case it's not already done in the config
        self.top_k = cfg.get("top_k", top_k * latent_ratio)
        self.num_experts = cfg.get("num_experts", num_experts * latent_ratio)

        self.latent_dim = cfg["emb_dim"] // latent_ratio
        self.routed_scaling_factor = cfg.get("routed_scaling_factor", routed_scaling_factor)
        self.shared_expert_hidden_dim = cfg.get("shared_expert_hidden_dim", shared_expert_hidden_dim)
        self.routed_expert_hidden_dim = cfg.get("routed_expert_hidden_dim", routed_expert_hidden_dim)

        self.routed_experts = nn.ModuleList([Expert(cfg, input_dim=self.latent_dim) for _ in range(self.num_experts)])
        self.shared_expert = Expert(cfg, hidden_dim=self.shared_expert_hidden_dim)
        self.gate = nn.Linear(cfg["emb_dim"], self.num_experts, bias=False, dtype=cfg["dtype"])

        # bias for Aux-Loss free load balancing (they used this loss from DeepSeek, see README.md)
        self.register_buffer("biases", torch.zeros(self.num_experts))
        self.bias_update_rate = bias_update_rate

        # latent projection layers
        self.down_proj = nn.Linear(cfg["emb_dim"], self.latent_dim, bias=False, dtype=cfg["dtype"])
        self.up_proj = nn.Linear(self.latent_dim, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        b, s, emb_dim = x.shape
        x_2d = x.view(-1, emb_dim)
        output = self.shared_expert(x).view(-1, emb_dim)  # preallocating with already shared expert output

        x_latent = self.down_proj(x_2d)
        # since dims aren't the same now, we can't add to the output tensor directly
        # we will accumulate the routed expert outputs in this tensor and then upscale once at the end
        routed_latent_output = torch.zeros_like(x_latent)

        # gating
        gate_logits = self.gate(x_2d)  # shape (b*s, num_routed)
        gate_probas = torch.sigmoid(gate_logits)
        biased_probas = gate_probas + self.biases  # they keep the biases for inference too

        topk_idxs = biased_probas.topk(self.top_k, dim=-1)[1]  # shape (b*s, topk)
        # retrieving topk probas (unbiased), normalizing to topk range and upscale probas/weights
        topk_probas = torch.gather(gate_probas, dim=-1, index=topk_idxs)
        topk_probas /= topk_probas.sum(dim=-1, keepdim=True)
        topk_probas *= self.routed_scaling_factor

        expert_mask = torch.nn.functional.one_hot(topk_idxs, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit_count = expert_mask.sum(dim=(-1, -2))
        expert_hit_idx = torch.where(expert_hit_count > 0)[0]

        # dispatching
        for idx in expert_hit_idx:
            expert_assignment = expert_mask[idx]  # shape (topk, b*s)
            topk_pos, token_idx = torch.where(expert_assignment)

            selected_tokens = x_latent[token_idx]  # shape (num_selected_tokens, latent_dim)
            selected_weights = topk_probas[token_idx, topk_pos].unsqueeze(-1)  # shape (num_selected_tokens, 1)

            expert_output = self.routed_experts[idx](selected_tokens) * selected_weights
            routed_latent_output.index_add_(dim=0, index=token_idx, source=expert_output)

        # update biases with the violation error
        if self.training:
            counts = torch.bincount(topk_idxs.view(-1), minlength=self.num_experts).float()  # float for mean()
            vio_error = counts.mean() - counts
            self.biases += self.bias_update_rate * vio_error.sign()

        routed_output = self.up_proj(routed_latent_output)
        output += routed_output
        output = output.view(b, s, emb_dim)

        return output


# quick inline test
if __name__ == "__main__":
    torch.manual_seed(123)

    cfg = {"emb_dim": 512, "moe_hidden_dim": 1024, "dtype": torch.bfloat16}
    model = LatentMoE(cfg, top_k=2, num_experts=4, latent_ratio=4)

    x = torch.randn(2, 10, 512).to(cfg["dtype"])
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(model.latent_dim)  # should be emb_dim / latent_ratio
    print(model.num_experts, model.top_k)  # should be num_experts * latent_ratio and top_k * latent_ratio
    print(f"Output shape: {output.shape}")
    print(output)
