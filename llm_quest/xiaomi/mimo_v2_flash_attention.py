# NOTE:
# The filename `mimo_v2_flash_attention.py` can be misleading.
# In the repo architecture files are prefixed by the name of the model, which is MiMo-V2-Flash here.
# We are not using "flash attention V2 (FA2)" library from https://github.com/Dao-AILab/flash-attention
# Even though it would be the best way to really leverage SWA benefits.
#
# For SWA here, we use the naive way, by simply applying a SWA causal mask to the global attention, so still O(L²)

import torch
import torch.nn as nn

from llm_quest.common.rope import RoPE
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm


class GroupedQueryAttentionWithSink(nn.Module):
    """
    A standard GQA with a learnable attention sink scalar/bias for the softmax calculation, as seen in gpt-oss
    https://arxiv.org/abs/2508.10925

    Values head dimension is decoupled from the QK head dimension, similar to Qwen3-Next in GDN.
    The attention sink is only applied for SWA layers, not for GA layers.

    Args:
        d_in (int): Input embedding dimension
        num_heads (int): Number of attention heads (for queries)
        num_kv_groups (int): Number of key-value groups, inferred from the cfg: SWA uses a different number of KV
        groups than GA.
        head_dim (int): Head dimension for Q and K
        value_head_dim (int, optional): Head dimension for V. If None, defaults to head_dim.
        dtype (torch.dtype, optional): Data type for weights
        is_sliding_window (bool): Whether this is a sliding window attention layer. If True, attention sink is applied.
    """

    def __init__(
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim,
        value_head_dim=None,
        dtype=None,
        is_sliding_window=False,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim if value_head_dim is not None else head_dim
        # output dimension input is based on value head now since QK heads dim != V heads dim
        self.d_out = num_heads * self.value_head_dim
        self.num_kv_groups = num_kv_groups
        self.num_repeat = num_heads // num_kv_groups
        self.att_scaling = head_dim**-0.5
        self.is_sliding_window = is_sliding_window

        # Q and K use head_dim
        qk_out_dim = num_heads * head_dim
        self.w_queries = nn.Linear(d_in, qk_out_dim, bias=False, dtype=dtype)
        self.w_keys = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        # V uses value_head_dim
        self.w_values = nn.Linear(d_in, num_kv_groups * self.value_head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm = PytorchRMSNorm(head_dim, dtype=dtype)
        self.k_norm = PytorchRMSNorm(head_dim, dtype=dtype)

        # Attention sink bias: learnable scalar per head
        if is_sliding_window:
            self.sink = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(num_heads,), dtype=dtype))

    def forward(self, x, mask, cos, sin, attn_mask=None):
        """
        Args:
            x: (b, seq_len, d_in)
            mask: (seq_len, seq_len) Base causal mask from GlobalBuffers
            cos, sin: RoPE embeddings, will be different for SWA and GA (since they have a different base RoPE)
            attn_mask: (b, seq_len) Optional padding mask (True = real token)
        """
        b, seq_len, d_in = x.shape

        # Project, reshape, and transpose Q, K, V (already done multiple times in the repo, chaining here)
        queries = self.w_queries(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.w_keys(x).view(b, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.w_values(x).view(b, seq_len, self.num_kv_groups, self.value_head_dim).transpose(1, 2)

        # QK Norm (before RoPE)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # partial rotation (first 64 dims in their case, which is about 33% of the head dim)
        queries = RoPE.apply(queries, cos, sin)
        keys = RoPE.apply(keys, cos, sin)

        # repeat K&V for GQA
        keys = keys.repeat_interleave(self.num_repeat, dim=1)
        values = values.repeat_interleave(self.num_repeat, dim=1)

        att_scores = (queries @ keys.mT) * self.att_scaling

        current_mask = mask[:seq_len, :seq_len]

        # apply attn/padding mask if provided
        if attn_mask is not None:
            combined_mask = current_mask.unsqueeze(0).unsqueeze(0) | ~attn_mask.unsqueeze(1).unsqueeze(1)
        else:
            combined_mask = current_mask

        att_scores.masked_fill_(combined_mask, -torch.inf)

        if self.is_sliding_window:
            # reshape sink from (num_heads) to (1, num_heads, 1, 1) and expand (as concat doesn't auto-broadcast)
            sink_broadcast = self.sink.view(1, self.num_heads, 1, 1).expand(b, -1, seq_len, -1)
            att_scores = torch.cat(
                [att_scores, sink_broadcast], dim=-1
            )  # adding sinks to attn scores only for the softmax

        # Optional:
        # @ArthurZucker added an extra (on top of the softmax doing it internally) max subtraction trick here before
        # passing to softmax, in bf16/fp16 cases, to prevent intermediate overflows.
        # att_scores = att_scores - att_scores.amax(dim=-1, keepdim=True)

        att_weights = torch.softmax(att_scores, dim=-1)

        if self.is_sliding_window:
            att_weights = att_weights[..., :-1]  # removing back the sinks for proper matmul with Values

        out = att_weights @ values
        out = out.transpose(1, 2).contiguous().view(b, seq_len, self.d_out)

        return self.out_proj(out)
