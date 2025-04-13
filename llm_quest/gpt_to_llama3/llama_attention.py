import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quest.rope import RoPE


# - Implementing relative+absolute positional embeddings with RoPE (instead of GPT2 absolute positional embeddings)
# - removing dropout
# - adding dtype setting
# - converting MHA → GQA


class GroupedQueryAttention(nn.Module):
    """
    GQA class, sharing key and value matrices between groups of query heads, reducing memory and computational cost.
    Modified version based on MultiHeadAttention class from gpt_attention.py

    Args:
        d_in (int): Input embedding dimension
        d_out (int): Output embedding dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        num_kv_groups (int): Number of key-value groups (must divide num_heads)
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.

    """

    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        num_kv_groups,
        dtype=None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // self.num_heads
        self.att_scaling = self.head_dim**-0.5
        self.num_kv_groups = num_kv_groups  # (if 1 = MQA, if num_heads = MHA, 1 < GQA < num_heads)
        self.num_repeat = self.num_heads // self.num_kv_groups
        self.w_queries = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # K and V projected onto num_kv_groups * head_dim, parameter efficiency of GQA
        self.w_keys = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_values = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)  # optional additional learnable params for the output

    def forward(self, x, mask, cos, sin):
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
        # need to duplicate "num_repeat" time K and V n_heads to match Q n_heads for matmul
        # ex: Q 2,10,6,5 !@ (2,2,6,5).T →  (*5 num_repeat=10/2) → Q @ (2,10,6,5).T
        # in our ex, since we are duping K heads (1st head 5 times, 2nd 5 times), each group of 5 query heads will
        # attend to the same duped keys (Q1@K1, ..., Q5@K1, Q6@K2,... Q10@K2), effectively grouping the attention
        keys = keys.repeat_interleave(self.num_repeat, dim=1)
        values = values.repeat_interleave(self.num_repeat, dim=1)
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
