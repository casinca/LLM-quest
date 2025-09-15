import torch
import torch.nn as nn

from llm_quest.gpt_to_llama3.llama_attention import GroupedQueryAttention

# add: - RMSNorm
#      - SwiGlu
#      - dtype setting
#
# remove: - dropout


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

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ scale factor (linear coeff)

    def forward(self, x):
        norm_x = x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps)
        return self.scale * norm_x


class SiLU(nn.Module):
    """
    This class implements the Sigmoid Linear Unit (SiLU) activation function.

    SiLU is defined as:
    SiLU(x) = x * σ(x)
    where σ(x) is the sigmoid function: 1/(1 + e^(-x))

    This activation function is also known as "swish" and is used as part of the SwiGLU
    gated activation in the Llama architecture's FFN, replacing the GELU activation from GPT.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * 1 / (1 + torch.exp(-x))


class FFN(nn.Module):
    """
    This class implements a Feed Forward Neural Network with SwiGLU activation.
    SwiGLU(x) = (xW + b) ⊗ SiLU(xV + c) then we project back onto the emb space with a 3rd linear transform.
    implementation applies SiLU to the first branch and not the gate.

    The gate linear transform is used for a bilinear transformation with the output of the 1st layer in order to scale
    (eg, amplify or dampens) the hidden states before projecting back onto the embedding space.

    The network consists of three linear layers (with no biases) and a SiLU activation function,
    structured as follows:
    x ⇉ (Linear1 → SiLU) * (Linear_gate) → Linear2 → output

    Args:
        cfg (dict): Config dictionary containing:
            - emb_dim (int): Input/output embedding dimension
            - hidden_dim (int): Hidden layer dimension
            - dtype (torch.dtype): Dtype of the weights, to change precision
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.lin1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.silu_activ = SiLU()
        self.lin_gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.lin2 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x1 = self.lin1(x)
        x1 = self.silu_activ(x1)
        x2 = self.lin_gate(x)

        return self.lin2(x1 * x2)


class TransformerBlock(nn.Module):
    """
    Implements a complete Transformer block

    This block consists of the following components:
    1. Grouped Query Attention
    2. RMS Normalization (pre-normalization)
    3. Feed-Forward Neural Network with SwiGLU
    4. Residual connections

    Args:
        cfg (dict): Config dict containing hyperparams:
            - emb_dim (int): Embedding dimension
            - context_length (int): Context length for attention
            - n_heads (int): Number of attention heads
            - num_kv_groups (int): Number of key-value groups for GQA
            - dtype (torch.dtype): Dtype of the weights, to change precision
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["num_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.norm_1 = RMSNorm(cfg["emb_dim"])
        self.norm_2 = RMSNorm(cfg["emb_dim"])
        self.ffn = FFN(cfg)

    def forward(self, x, mask, cos, sin):
        # Pre-norm arch
        residual = x
        x = self.norm_1(x)
        x = self.att(x, mask, cos, sin)
        x = x + residual

        residual = x
        x = self.norm_2(x)
        x = self.ffn(x)
        x = x + residual

        return x
