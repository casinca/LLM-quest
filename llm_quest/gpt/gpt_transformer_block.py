import torch
import torch.nn as nn

from llm_quest.gpt.gpt_attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """
    This class implements the Layer "Normalization" (more like a standardization...).

    It is an affine transformation of a Z-score of our input x, affine on purpose to make it learnable for the
    model.
    layer_norm = γ * zscore + β. With zscore as (x - μ) / (σ + ε).
    - ε small const to avoid div by 0,
    - σ std dev
    - μ mean
    The operation is applied to each token independently (each emb dim)
    The learnable coeff γ and intercept β are called scale and shift parameters (default values: 1 and 0 → y=x)

    Recentering and rescaling the embeddings helps the model learn better/converge faster, consistency reduces risk
    of gradient exploding or vanishing, also helps a bit with internal covariate shift.

    Args:
        emb_dim (int): The dimension of the tokens embeddings.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # scale factor (γ linear coeff)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # shift factor (which is the added intercept β)

    def forward(self, x):
        std_dev = torch.std(x, dim=-1, keepdim=True, unbiased=False)  # unbiased = Bessel's correction for variance /n-1
        mean = x.mean(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std_dev + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) smooth activation function.
    This activation function is defined as:
    GELU(x) = x * Φ(x)
    Where Φ(x) is the CDF of the standard normal distribution.

    The function can be approximated as:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    but I'm using it with the error function version from the paper, using torch.erf():
    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 1 / 2 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2))))


class FFN(nn.Module):
    """
    This class implements a small Feed Forward Neural Network.

    2 linear layers (input-hidden & hidden-output) structured as follows:
    input layer → hidden layer → (acti:GELU)→ output layer
    The hidden layer is 4* the input dim size

    Args:
        cfg (dict): Config dict containing the embedding dim.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Implements a complete Transformer block with self-attention.

    This block consists of the following components:
    1. Multi-Head Self-Attention
    2. Layer Normalization
    3. Feed Forward Neural Network
    4. Residual connections and dropout for regularization

    Args:
        cfg (dict): Config dict containing hyperparams:
            - emb_dim (int): Embedding dimension
            - drop_rate (float): Dropout rate
            - context_length (int): Context length for attention
            - n_heads (int): Number of attention heads
            - qkv_bias (bool): Whether to use bias in query, key, and value projections
    """

    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"],
            ctx_len=cfg["context_length"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ln_1 = LayerNorm(cfg["emb_dim"])
        self.ln_2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FFN(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attn_mask=None):
        """
        This is a pre-LN arch, contrary to the original paper on transformers (which is post-LN)
        Somehow GPT paper has a post-LN fig.1 but OpenAI impl is pre-LN

        Args:
            x: Input tensor of shape (b, seq_len, emb_dim)
            attn_mask: Optional attention mask of shape (b, seq_len)
        """
        residual = x
        x = self.ln_1(x)
        x = self.att(x, attn_mask)
        x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.ln_2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x


# testing code
# if __name__ == "__main__":
#    torch.manual_seed(123)
#    batch_example = torch.randn(2, 5)
#    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
#    out = layer(batch_example)
#    print(out)
#
#    print(batch_example)
#    layer_nm = LayerNorm(batch_example.shape[-1])
#    ln_out = layer_nm(batch_example)
#    print(ln_out)
#    print(ln_out.std(-1, keepdim=True, unbiased=False), ln_out.mean(-1, keepdim=True))
#
#    cfg = {"emb_dim": 768}
#    ffn = FFN(cfg)
#    rand_x = torch.rand(2, 4, 768)
#    print(ffn(rand_x).shape)
#
#    trf = TransformerBlock(GPT_CONFIG_124M)
#    output = trf(rand_x)
#    print(output.shape)
