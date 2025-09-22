import math

import torch
import torch.nn as nn

from llm_quest.llama3_to_gemma3.gemma3_attention import GroupedQueryAttention

# add: GeGLU
# remove: SwiGLU


# RMSNorm is used instead of LayerNorm for the "normalization" of the embeddings
# RMSNorm only focuses on re-scaling invariance, doesn't have an intercept (shift) nor recenter to a μ=0.
class RMSNorm(nn.Module):
    """
    This class implements Root Mean Square "Normalization".

    RMSNorm is a simplified version of LayerNorm that only focuses on re-scaling invariance.
    RMSNorm = γ * x / (RMS(x) + ε)
    ε small value to avoid div by 0 (same as LayerNorm)
    The only learnable parameter is the scale factor γ which multiplies the normalized input.

    Args:
        emb_dim (int): The dimension of the embeddings to "normalize" over.
    """

    def __init__(self, emb_dim, dtype=None):
        super().__init__()
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(emb_dim, dtype=dtype))  # γ scale factor (linear coeff)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        norm_x = x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps)  # sensitive RMS part in fp32
        return self.scale * norm_x.to(input_dtype)  # partial cast (full cast: (self.scale * norm_x).to(input_dtype))


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) smooth activation function.
    This activation function is defined as:
    GELU(x) = x * Φ(x)
    Where Φ(x) is the CDF of the standard normal distribution.

    The function can be approximated as:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    but I'm using it with the error function version from the paper, using torch.erf():
    GELU(x) = x * 1/2 * (1 + erf(x / sqrt(2)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class FFN(nn.Module):
    """
    This class implements a Feed Forward Neural Network with GeGLU activation.
    GeGLU(x) = (xW + b) ⊗ GELU(xV + c) then we project back onto the emb space with a 3rd linear transform.
    (note: +b and +c if bias=True)
    which gives FFN_GeGLU(x, W, V, W2) = (GELU(xW) ⊗ xV)W2

    The linear gate transform is used for a bilinear transformation with the output of the 1st layer in order to scale
    (eg, amplify or dampens) the hidden states before projecting back onto the embedding space (with the last linear
    layer).

    So the network consists of three linear layers and a GELU activation function structured as
    follows:

    #
    #       ┌─────────┐
    # x ───►Linear_gate ─────► ┌──────┐
    #       └─────────┘          │ GELU │───┐
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
        self.gelu_activ = GELU()
        self.lin2 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.gelu_activ(x_gate)
        x1 = self.lin1(x)

        return self.lin2(x1 * x_gate)


class TransformerBlock(nn.Module):
    """
    Implements a complete Transformer block

    This block consists of the following components:
    1. Grouped Query Attention
    2. RMS Normalization (pre-normalization)
    3. Feed-Forward Neural Network with GeGLU
    4. Residual connections

    Args:
        cfg (dict): Config dict containing hyperparams:
            - emb_dim (int): Embedding dimension
            - context_length (int): Context length for attention
            - n_heads (int): Number of attention heads
            - num_kv_groups (int): Number of key-value groups for GQA
            - window_size (int): Window size for SWA
            - local_global_att_ratio (int): Local-global attention ratio
            - dtype (torch.dtype): Dtype of the weights, to change precision
    """

    def __init__(self, cfg, layer):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["num_kv_groups"],
            window_size=cfg["window_size"],
            layer_id=layer,
            dtype=cfg["dtype"],
            local_global_att_ratio=cfg["local_global_att_ratio"],
        )
        self.norm_1 = RMSNorm(cfg["emb_dim"])
        self.norm_2 = RMSNorm(cfg["emb_dim"])
        self.ffn = FFN(cfg)

    def forward(self, x, mask, cos, sin, swa_mask=None):

        residual = x
        x = self.norm_1(x)
        x = self.att(x, mask, cos, sin, swa_mask)
        x = x + residual
        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = x + residual

        return x
