import torch.nn as nn

from llm_quest.moe.qwen3_moe import Qwen3MoE
from llm_quest.qwen3.qwen3_attention import GroupedQueryAttention, PytorchRMSNorm


class FFN(nn.Module):
    """
    This class implements a Feed Forward Neural Network with SwiGLU activation.
    SwiGLU(x) = (xW + b) ⊗ SiLU(xV + c) then we project back onto the emb space with a 3rd linear transform.
    which gives FFN_SwiGLU(x, W, V, W2) = (Swish(xW) ⊗ xV)W2

    note: +b and +c if bias=True

    The linear gate transform is used for a bilinear transformation with the output of the 1st layer in order to scale
    (eg, amplify or dampens) the hidden states before projecting back onto the embedding space (with the last linear
    layer).

    So the network consists of three linear layers and a SiLU activation function structured as
    follows:

    #
    #       ┌─────────┐
    # x ───►Linear_gate ─────► ┌──────┐
    #       └─────────┘          │ SiLU │───┐
    #                            └──────┘   │
    #                                      ▼
    #                                      * (Elem-wise mult) ─────► ┌─────────┐
    #                                      ▲                         │ Linear2 │───► output
    #       ┌───────────┐                  │                          └─────────┘
    # x ───►  Linear1  │──────────────────┘
    #       └───────────┘

    Args:
        cfg (dict): Config dictionary containing:
            - emb_dim (int): Input/output embedding dimension
            - hidden_dim (int): Hidden layer dimension
            - dtype (torch.dtype): Dtype of the weights, to change precision
    """

    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.lin_gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.silu_activ = nn.functional.silu  # Pytorch's built-in
        self.lin2 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.silu_activ(x_gate)
        x1 = self.lin1(x)

        return self.lin2(x1 * x_gate)


class TransformerBlock(nn.Module):
    """
    Implements a complete Qwen3 Transformer block

    This block consists of the following components:
    1. Grouped Query Attention with QK-Norm
    2. RMS Normalization (pre-normalization)
    3. Feed-Forward Neural Network with SwiGLU
    4. Residual connections

    Args:
        cfg (dict): Config dict containing hyperparams:
            - emb_dim (int): Embedding dimension
            - context_length (int): Context length for attention
            - n_heads (int): Number of attention heads
            - num_kv_groups (int): Number of key-value groups for GQA
            - head_dim (int): Head dimension for GQA
            - dtype (torch.dtype): Dtype of the weights, to change precision
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["num_kv_groups"],
            head_dim=cfg["head_dim"],
            dtype=cfg["dtype"],
        )
        self.norm1 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm2 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.ffn = FFN(cfg)

    def forward(self, x, mask, cos, sin):
        # Pre-norm arch
        residual = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class MoETransformerBlock(nn.Module):
    """
    Implements a Qwen3 Transformer block with Mixture of Experts.
    Same as Dense transformer block, but with MoE instead of FFN.
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["num_kv_groups"],
            head_dim=cfg["head_dim"],
            dtype=cfg["dtype"],
        )
        self.norm_1 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm_2 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.moe = Qwen3MoE(cfg=cfg, training=self.training)

    def forward(self, x, mask, cos, sin):
        # Pre-normalization architecture
        residual = x
        x = self.norm_1(x)
        x = self.att(x, mask, cos, sin)
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.moe(x)  # Use MoE instead of regular FFN
        x = x + residual

        return x
