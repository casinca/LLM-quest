import math

import torch
import torch.nn as nn

from llm_quest.vit.vit_attention import ViTMultiHeadAttention

# Nothing changes here, same as GPT's TransformerBlock, we just replace causal MHA with ViT's MHA attending to all
# patches (no autoregression)


class LayerNorm(nn.Module):
    """
    Layer "Normalization" for ViT.
    Same as used in GPT.

    Args:
        emb_dim (int): The dimension of the embeddings.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ (scale)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β (shift)

    def forward(self, x):
        std_dev = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        mean = x.mean(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std_dev + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Same as used in GPT and recommended in the ViT paper.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class FFN(nn.Module):
    """
    Feed Forward Neural Network.

    Again here, same structure as GPT's FFN:
    input_dim → hidden_dim (4x) → GELU → output_dim

    Args:
        cfg (dict): Config dict containing the embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class ViTTransformerBlock(nn.Module):
    """
    Vision Transformer Block (Encoder-only).

    This is adapted from the GPT TransformerBlock but uses ViT attention
    (no causal masking) and follows the ViT paper's arch:
    1. Layer Norm → MHA → Residual
    2. Layer Norm → FFN → Residual

    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension
            - drop_rate (float): Dropout rate
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in QKV projections
    """

    def __init__(self, cfg):
        super().__init__()

        self.att = ViTMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ln_1 = LayerNorm(cfg["emb_dim"])
        self.ln_2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FFN(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass using pre-LayerNorm architecture (same as GPT).

        Args:
            x: Input tensor of shape (b, seq_len, emb_dim)
                where seq_len = num_patches + 1 (for cls token)

        Returns:
            Output tensor of same shape as input
        """
        # 1st block (Self-attention block aka Z'_l eq.2 in the paper)
        residual = x
        x = self.ln_1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + residual

        # 2nd block (FFN block aka Z_l eq.3 in the paper)
        residual = x
        x = self.ln_2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x


# Testing code
if __name__ == "__main__":
    torch.manual_seed(123)

    # dummy ViT config (similar to ViT-Base)
    vit_cfg = {
        "emb_dim": 768,
        "drop_rate": 0.1,
        "n_heads": 12,
        "qkv_bias": False,
    }

    vit_block = ViTTransformerBlock(vit_cfg)

    # Ex input: batch_size=2, seq_len=5 (4 patches + 1 cls token), emb_dim=768
    x = torch.randn(2, 5, 768)

    output = vit_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
