import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quest.rope import RoPE


# - add:
#   - SWA
#   - adapt GQA for SWA
#   - LayerNorm for QK normalization (copy from dir /gpt/gpt_transformer_block.py)


class LayerNorm(nn.Module):
    """
    This class implements the Layer "Normalization".

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


def apply_sliding_window_attention(queries, keys, values, window_size, swa_mask=None):
    """
    Sliding Window Attention.

    This function computes attention scores within a fixed-size window for each query token.
    It uses padding and stride tricks to create sliding windows for keys and values.
    Rest is classic 2017 attention.

    Args:
        queries (torch.Tensor): Query tensor of shape (b, num_heads, seq_len, head_dim).
        keys (torch.Tensor): Key tensor of shape (b, num_heads, seq_len, head_dim).
        values (torch.Tensor): Value tensor of shape (b, num_heads, seq_len, head_dim).
        window_size (int): The size of the sliding attention window.
        swa_mask (torch.Tensor, optional): Precomputed sliding window attention mask
            of shape (seq_len, window_size). Defaults to None, but should be provided
            for causal masking.

    Returns:
        torch.Tensor: The context tensor after applying sliding window attention,
                    of shape (b, num_heads, seq_len, head_dim).
    """

    # keys and values should already be pre-shaped as:
    b, num_heads, seq_len, head_dim = keys.shape

    # Padding for keys and values:
    # it's needed for maintaining the fixed window size for the first tokens < window_size
    # ex w=3 → [PAD,PAD,TOK1], [PAD,TOK1,TOK2] and not [TOK1], [TOK1, TOK2]
    pad_size = window_size - 1
    # similar to CV with CNN etc..., here we pad the top, ie before the first token vector
    # shape (b, num_heads, seq_len + pad_size, head_dim)
    padded_keys = F.pad(keys, (0, 0, pad_size, 0))  # (left, right, top, bottom)
    padded_values = F.pad(values, (0, 0, pad_size, 0))

    # stride is the jump necessary to go from one element to the next one in the specified dimension
    # ex with a matrix of shape (2,5), to jump from the 3rd elem of the 1st vector to the 3rd elem of the 2nd vector,
    # we need to jump 5 positions.
    # strides for each dim:
    b_stride, h_stride, s_stride, d_stride = padded_keys.stride()
    new_stride = (
        b_stride,
        h_stride,
        s_stride,  # seq_len stride (1 position per window step)
        s_stride,  # move by stride_s for both seq_len and window_size dim (1 position per element)
        d_stride,
    )
    # the target shape to go with our stride, 5D tensor:
    target_shape = (b, num_heads, seq_len, window_size, head_dim)

    keys_windows = padded_keys.as_strided(size=target_shape, stride=new_stride)
    values_windows = padded_values.as_strided(size=target_shape, stride=new_stride)

    # Attention:
    # need to match query shape with keys for matmul:
    #  (b, num_heads, seq_len, head_dim) -> (b, num_heads, seq_len, 1, head_dim)
    # and keys^T -> (b, num_heads, seq_len, head_dim, window_size)
    att_scores = queries.unsqueeze(3) @ keys_windows.mT

    # Causal mask:
    # same as padding, need to take care of tokens < window_size.
    # for a position k within the window, we want to mask all positions where k < w - 1 - i
    # with i the position within the seq length.
    # ex: w=3, seq_len=5
    # start i=0, mask k < 3-1-0 = 2  so for the first token [True, True, False] only attend to itself
    # i=1, mask k < 1 so for the second token [True, False, False], i=2 will have full window available, etc...

    # ##### Moved to GlobalBuffers, for reference: ######
    # k_range = torch.arange(window_size, device=queries.device)
    # i_range = torch.arange(seq_len, device=queries.device).unsqueeze(-1)
    # swa_mask = k_range < (window_size - 1 - i_range)  # shape (seq_len, window_size) Pytorch will auto-broadcast

    # rest is classic GQA:
    # scaling, masking(w/ swa_mask(ctx_len) up to seq_len) and softmax'ing
    scaled_att_scores = att_scores.squeeze(3) * head_dim**-0.5
    scaled_att_scores.masked_fill_(swa_mask[:seq_len, :], -torch.inf)
    att_weights = F.softmax(scaled_att_scores, dim=-1)

    ctx_tensor = att_weights.unsqueeze(3) @ values_windows

    return ctx_tensor.squeeze(3)


class GroupedQueryAttention(nn.Module):
    """
    GQA class, sharing key and value matrices between groups of query heads, reducing memory and computational cost.
    Modified version based on MultiHeadAttention class from gpt_attention.py

    Args:
        d_in (int): Input embedding dimension
        d_out (int): Output embedding dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        num_kv_groups (int): Number of key-value groups (must divide num_heads)
        window_size (int): Size of the sliding window
        layer_id (int): Layer index
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.
        local_global_att_ratio (int, optional): Ratio of local attention to global attention. Defaults to 5. (if 0 =
        full global, if num_layers = full local)

    """

    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        num_kv_groups,
        window_size,
        layer_id,
        dtype=None,
        local_global_att_ratio=5,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // self.num_heads
        self.num_kv_groups = num_kv_groups  # (if 1 = MQA, if num_heads = MHA, 1 < GQA < num_heads)
        self.num_repeat = self.num_heads // self.num_kv_groups
        self.att_scaling = self.head_dim**-0.5
        self.w_queries = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # K and V projected onto num_kv_groups * head_dim, parameter efficiency of GQA
        self.w_keys = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_values = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)  # optional additional learnable params for the output
        self.window_size = window_size
        self.layer_id = layer_id
        self.lg_ratio = local_global_att_ratio + 1  # +1 for 0 indexing fix
        self.q_ln = LayerNorm(self.head_dim)
        self.k_ln = LayerNorm(self.head_dim)

    def forward(self, x, mask, cos, sin, swa_mask=None):
        queries = self.w_queries(x)  # shape (b, s, d_out)
        keys = self.w_keys(x)  # K and V shapes (b, s, num_kv_groups * head_dim)
        values = self.w_values(x)

        b, seq_len, d_in = x.shape
        # reshaping at the d_out dim our 3D tensors by the num of heads(Q) or groups(KV), reshaped, is a 4D tensor:
        # (b, s, num of heads(Q) or groups(KV), head dim)
        # ex: Q 2,6,50 → (10heads) → 2,6,10,5 and KV with kvgroups=2: 2,6,10(2*5) → 2,6,2,5
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_kv_groups, -1)
        values = values.view(b, seq_len, self.num_kv_groups, -1)

        # att scores
        # transposing first, num_head and seq_len for correct matmul within each head, ex: Q 2,6,10,5 → 2,10,6,5
        queries = torch.transpose(queries, 1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # applying pos embedding with RopE
        queries = RoPE.apply(queries, cos, sin)
        keys = RoPE.apply(keys, cos, sin)
        # QK Norm (after RoPE, since we need og vectors for rotating)
        queries = self.q_ln(queries)
        keys = self.k_ln(keys)
        # need to duplicate "num_repeat" time K and V n_heads to match Q n_heads for matmul
        # ex: Q 2,10,6,5 !@ (2,2,6,5).T →  (*5 num_repeat=10/2) → Q @ (2,10,6,5).T
        # in our ex, since we are duping K heads (1st head 5 times, 2nd 5 times), each group of 5 query heads will
        # attend to the same duped keys (Q1@K1, ..., Q5@K1, Q6@K2,... Q10@K2), effectively grouping the attention
        keys = keys.repeat_interleave(self.num_repeat, dim=1)
        values = values.repeat_interleave(self.num_repeat, dim=1)

        # Alternating SWA depending on the layer num
        if self.window_size > 0 and (self.layer_id + 1) % self.lg_ratio:
            ctx_tensor = apply_sliding_window_attention(
                queries, keys, values, window_size=self.window_size, swa_mask=swa_mask
            )

        else:
            att_scores = queries @ keys.mT  # shape (b, num_heads, seq_len, seq_len)
            # mask up to seq length/num of tokens
            current_mask = mask.bool()[:seq_len, :seq_len]
            # scaling by √(head_dim)
            scaled_att_scores = att_scores * self.att_scaling
            # masking in place and normalizing with softmax
            scaled_att_scores.masked_fill_(current_mask, -torch.inf)
            att_weights = F.softmax(scaled_att_scores, dim=-1)

            ctx_tensor = att_weights @ values

        # transposing one last time to get back our initial split shape
        # (batch, num_heads, seq_len, head_dim) → (batch, seq len, num_heads, head_dim)
        ctx_tensor = ctx_tensor.transpose(1, 2)
        # merging back to our initial 3D tensor shape (b, seq len, d_out)
        # note: contiguous() since transpose doesn't return a contiguous tensor, and necessary for view()
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)
        # passing context vectors to a final linear transform for output normalization & additional learnable params
        ctx_tensor = self.out_proj(ctx_tensor)

        return ctx_tensor


# quick test
if __name__ == "__main__":
    from gemma3_model import GlobalBuffers

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

    swa_mask = GlobalBuffers().get_swa_buffers(ctx_len, 3)
    mask, cos, sin = GlobalBuffers().get_buffers(ctx_len, 10_000, 2)

    d_in = inputs.shape[-1]
    d_out = 12
    multi_head = GroupedQueryAttention(
        d_in,
        d_out,
        num_heads=6,
        num_kv_groups=2,
        window_size=3,
        layer_id=0,
    )
    print(multi_head(input_batch, mask, cos, sin, swa_mask))
