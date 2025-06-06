import torch
import torch.nn as nn

# The Attention class is the same as MultiHeadAttention in `gpt_attention.py` but we remove causal masking to switch
# from decoder to encoder architecture needed for ViT


class ViTMultiHeadAttention(nn.Module):
    """
    MHA module for Vision Transformer (encoder-only, no causal masking).

    This is adapted from the GPT MultiHeadAttention but removes causal masking since
    ViT processes all patches simultaneously without autoregressive dependencies.

    Args:
        d_in (int): Input embedding dimension
        d_out (int): Output embedding dimension (must be divisible by num_heads)
        dropout (float): Dropout probability
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to include bias terms in query/key/value projections.
                                Defaults to False.
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.att_scaling = self.head_dim**-0.5

        # QKV projections
        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout and output projection
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        """
        we remove causal mask (we also don't use attention masks, we expect images of WxH fixed size)

        args:
            x: (b, seq_len, d_in)

        Note: seq_len is actually = num_patches + 1 (for cls token)

        Returns:
            (b, seq_len, d_out)
        """
        b, seq_len, d_in = x.shape

        # project to queries, keys, values
        queries = self.w_queries(x)  # (b, seq_len, d_out)
        keys = self.w_keys(x)
        values = self.w_values(x)

        # reshape for multi-head attention
        # (b, seq_len, num_heads, head_dim)
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_heads, -1)
        values = values.view(b, seq_len, self.num_heads, -1)

        # transpose to (b, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute attention scores
        att_scores = queries @ keys.mT  # (b, num_heads, seq_len, seq_len)
        scaled_att_scores = att_scores * self.att_scaling

        # apply softmax (no causal masking for ViT!)
        att_weights = torch.softmax(scaled_att_scores, dim=-1)
        att_weights = self.dropout(att_weights)

        # apply attention to values
        ctx_tensor = att_weights @ values  # (b, num_heads, seq_len, head_dim)

        # transpose back and reshape
        ctx_tensor = ctx_tensor.transpose(1, 2)  # (b, seq_len, num_heads, head_dim)
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)

        # final output projection
        ctx_tensor = self.out_proj(ctx_tensor)

        return ctx_tensor


# Testing code
if __name__ == "__main__":
    torch.manual_seed(123)

    # Ex: 4 patches + 1 cls token = 5 total sequence length
    batch_size = 2
    seq_len = 5
    d_in = 768
    d_out = 768
    num_heads = 12

    # create some dummy input (simulating patch embeddings + cls token)
    x = torch.randn(batch_size, seq_len, d_in)

    vit_attention = ViTMultiHeadAttention(d_in, d_out, dropout=0.1, num_heads=num_heads)

    output = vit_attention(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
