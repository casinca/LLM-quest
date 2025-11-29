# implementation of Dynamic Tanh from:
#
# Transformers without Normalization
# Paper https://arxiv.org/abs/2503.10622
# Authors: Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu

import torch


class DyT(torch.nn.Module):
    """
    DyT (Dynamic Tanh) class is a shifting and scaling layer, without Normalization.
    It is an alternative to the common RMSNorm or LayerNorm.

    Args:
        emb_dim (int): The dimension of the embeddings to shift and scale
        alpha (float): The learnable scalar/coeff for scaling x inside the tanh function

    Note:
        - The starting dynamic coeff (α) is very sensitive for LLM training.
        Paper mentions: - inverse proportionality to model width/emb_dim
                        - higher α for Attention blocks, lower α for FFNs & out layer

    """

    def __init__(self, emb_dim, alpha=1.0):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.ones(emb_dim))  # γ scale factor (γ linear coeff)
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))  # shift factor ( β which is the added intercept)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))  # learnable scalar as a dynamic coeff for x

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.scale * x + self.shift
