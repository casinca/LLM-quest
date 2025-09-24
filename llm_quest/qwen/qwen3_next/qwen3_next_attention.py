import torch
import torch.nn as nn

from llm_quest.common.rope import RoPE

# compared to `Qwen3_attention.py`:
# We removed all QK Norm and use Zero-Centered RMSNorm instead
# The classic Attention block is now a Gated SDPA block (basically GQA with an extra sigmoid activated gate)
# GQA was implemented multiple times in the repo, for a change, using Pytorch's SDPA function
# TODO
# TODO pass only the cfg, retrieves args from it + include training flag
# TODO IMPORTANT adapt mask passing to avoid inverting everytime, curr is not inverted!


class ZeroCenteredRMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm used in Qwen3-Next
    This is a RMSNorm with 0 centered weights (instead of classic 1s initialization)

    To make the forward pass work, we add 1 to the weights to get the correct scaling back.
    This is a trick to better adapt L2 regularization with RMSNorm.

    Note: We also do the full forward in fp32

    Args:
        emb_dim (int): The dimension of the embeddings to "normalize" over.
        eps (float): The epsilon value to avoid division by zero.
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.
    """

    def __init__(self, emb_dim, eps=1e-6, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(emb_dim, dtype=dtype))  # 0 centered weights
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * rms * (1 + self.scale)).to(input_dtype)  # fullcast to fp32 before returning to input dtype


class GatedAttention(nn.Module):
    """
    Gated Scaled Dot Product Attention(SDPA) using GQA as described in Qwen3-Next blogpost.
    It's similar to `Qwen3_attention.py` but with an added sigmoid activated gate and Zero-Centered RMSNorm for QK
    normalization

    Using Pytorch's built-in SDPA function for Attention calc

    Args:
        d_in (int): Input embedding dimension
        num_heads (int): Number of attention heads
        head_dim (int): Head dimension
        num_kv_groups (int): Number of key-value groups (must divide num_heads)
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.
    """

    def __init__(self, d_in, num_heads, head_dim, num_kv_groups, p_dropout, training=True, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = self.num_heads * self.head_dim
        self.num_kv_groups = num_kv_groups
        self.num_repeat = self.num_heads // self.num_kv_groups
        self.p_dropout = p_dropout if training else 0.0

        self.w_queries = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.w_keys = nn.Linear(d_in, self.num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_values = nn.Linear(d_in, self.num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_gate = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, attn_mask=None):
        """
        args:
            x: (b, seq_len, d_in)
            attn_mask (optional): (b, seq_len) used for padding tokens
        """
        b, seq_len, d_in = x.shape

        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)
        gate_output = torch.sigmoid(self.w_gate(x))

        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_kv_groups, -1)
        values = values.view(b, seq_len, self.num_kv_groups, -1)

        queries = torch.transpose(queries, 1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # rotating only first 25% of the dimensions with RoPE
        queries = RoPE.apply(queries, cos, sin)
        keys = RoPE.apply(keys, cos, sin)

        curr_mask = mask[:seq_len, :seq_len]
        if attn_mask is not None:
            # reshape & combine masks (invert attn_mask to get True = padding)
            curr_mask = curr_mask.view(1, 1, seq_len, seq_len) | ~attn_mask.view(b, 1, 1, seq_len)

        # (auto backend, not wrapped in with sdpa_kernel for compatibility)
        ctx_tensor = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=curr_mask,
            is_causal=False,
            dropout_p=self.p_dropout,
            enable_gqa=True,
        )

        ctx_tensor = ctx_tensor.transpose(1, 2)
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)

        ctx_tensor = ctx_tensor * gate_output
        ctx_tensor = self.out_proj(ctx_tensor)

        return ctx_tensor


# quick test
if __name__ == "__main__":
    from llm_quest.common.buffers import GlobalBuffers

    torch.manual_seed(123)

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89, 0.15, 0.15],  # Your (x^1)
            [0.55, 0.87, 0.66, 0.87, 0.87],  # journey (x^2)
            [0.57, 0.85, 0.64, 0.85, 0.85],  # starts (x^3)
            [0.22, 0.58, 0.33, 0.58, 0.58],  # with (x^4)
            [0.77, 0.25, 0.10, 0.25, 0.25],  # one (x^5)
            [0.05, 0.80, 0.55, 0.80, 0.80],  # step (x^6)
        ]
    )

    input_batch = torch.stack((inputs, inputs), dim=0)
    # context length/ seq length (b, s, emb_dim)
    ctx_len = input_batch.shape[1]

    mask, cos, sin = GlobalBuffers().get_buffers(ctx_len, 10_000, 2)
    mask = ~mask  # for backward compatibility inverting (SPDA:Mask where False OUR:Mask where True)

    d_in = inputs.shape[-1]
    d_out = 12
    gsdpa = GatedAttention(
        d_in=d_in,
        num_heads=6,
        head_dim=2,
        num_kv_groups=2,
        p_dropout=0.0,
        training=True,
    )

    print(gsdpa(input_batch, mask, cos, sin))
