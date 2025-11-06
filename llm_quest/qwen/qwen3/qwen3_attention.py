import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quest.common.rope import RoPE


# NOTE RMSNorm has already been implemented, using Pytorch's RMSNorm for changing
# NOTE FOR REPRO:
# Can't partially cast Pytorch's RMSNorm to fp32 vs HF's impl (weights aren't promoted to fp32, only RMS part)
# if using pytorch RMSNorm, some prompts partially diverge
# if using our own fullcast to fp32, prompts match 100% of the time
# if using our own partial cast, some prompts partially diverge
# it's counterintuitive but maybe because of @use_kernel_forward_from_hub("RMSNorm") that overrides their partial cast
# with a fullcast?
# TODO:
# - use from scratch but more efficient, if as fast as Pytorch
# - put all normalizations in common and import from there
class PytorchRMSNorm(torch.nn.RMSNorm):
    """
    Wrapper of Pytorch's RMSNorm.
    """

    def __init__(self, emb_dim, eps=1e-6, dtype=None):
        super().__init__(emb_dim, eps=eps, dtype=dtype)

    def forward(self, x):
        input_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(input_dtype)  # fullcast to fp32 before returning to input dtype


class GroupedQueryAttention(nn.Module):
    """
    GQA class for Qwen3, sharing key and value matrices between groups of query heads,
    with QK-Norm applied after RoPE (for training stability and matching Qwen3 way).

    Also removed QKV bias (as mentioned in Qwen3 paper)

    Args:
        d_in (int): Input embedding dimension
        num_heads (int): Number of attention heads
        num_kv_groups (int): Number of key-value groups (must divide num_heads)
        head_dim (int): Head dimension
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.
    """

    def __init__(
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim,
        dtype=None,
        layer_idx=None,
    ):
        super().__init__()
        # (since head_dim is now a specific hparam no need for assert d_in (or d_out if = d_in) % num_heads == 0)
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        # head dim is now a specific hparam in Qwen3 config (not inferred from d_out)
        # therefore d_out is not necessarily the same as d_in (emb_dim) anymore
        self.d_out = self.num_heads * self.head_dim

        self.att_scaling = self.head_dim**-0.5
        self.num_kv_groups = num_kv_groups  # (if 1 = MQA, if num_heads = MHA, 1 < GQA < num_heads)
        self.num_repeat = self.num_heads // self.num_kv_groups

        # no bias in Qwen3 (removed from Qwen2)
        self.w_queries = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # K and V projected onto num_kv_groups * head_dim, parameter efficiency of GQA
        self.w_keys = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_values = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        self.q_norm = PytorchRMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = PytorchRMSNorm(self.head_dim, dtype=dtype)

    def forward(self, x, mask, cos, sin, attn_mask=None, kv_cache=None, position_ids=None):
        """
        args:
            x: (b, seq_len, d_in)
            mask: (b, seq_len) Causal mask (passed as True = future tokens/upper right triangle)
            attn_mask: (b, seq_len) Attention mask (passed as True = real tokens)
            kv_cache: KVCache instance/object
            position_ids: (b, s/1) (long tensor), containing the position of each token in the sequence
        """
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

        # transposing first, num_head and seq_len for correct matmul within each head, ex: Q 2,6,10,5 → 2,10,6,5
        queries = torch.transpose(queries, 1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # QK-Norm applied before RoPE (order is important for matching how training was done in Qwen3)
        # Why applied before RoPE? https://github.com/allenai/OLMo/issues/806
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # rotating features for positional information, with RoPE, after QK normalization
        queries = RoPE.apply(queries, cos, sin, position_ids)
        keys = RoPE.apply(keys, cos, sin, position_ids)

        if kv_cache is not None:
            keys, values = kv_cache.get_updated_cache(keys, values, self.layer_idx)

        # need to duplicate "num_repeat" time K and V n_heads to match Q n_heads for matmul
        keys = keys.repeat_interleave(self.num_repeat, dim=1)
        values = values.repeat_interleave(self.num_repeat, dim=1)
        att_scores = queries @ keys.mT  # shape (b, num_heads, seq_len, seq_len)

        # before retrieving K from the KVCache, k_seq_len = q_seq_len = 1. After retrieving K, k_seq_len = seq_len
        q_seq_len = queries.shape[2]
        k_seq_len = keys.shape[2]
        # masking: causal mask and attn_mask
        if k_seq_len > q_seq_len:  # imply KVCache is used
            q_start_pos = k_seq_len - q_seq_len  # q_seq_len should be 1 for classic NTP KVCache inference
            current_mask = mask[q_start_pos:k_seq_len, :k_seq_len]
        else:
            current_mask = mask[:q_seq_len, :k_seq_len]
        if attn_mask is not None:
            # causal shape (s, s) → (1, 1, q_seq_len, k_seq_len), attn_mask shape (b, s) → (b, 1, 1, k_seq_len)
            current_mask = current_mask.unsqueeze(0).unsqueeze(0) | ~attn_mask.unsqueeze(1).unsqueeze(1)

        scaled_att_scores = att_scores * self.att_scaling
        mask_value = torch.finfo(scaled_att_scores.dtype).min / 2
        scaled_att_scores.masked_fill_(current_mask, mask_value)
        att_weights = F.softmax(scaled_att_scores, dim=-1)

        ctx_tensor = att_weights @ values
        # (batch, num_heads, seq_len, head_dim) → (batch, seq len, num_heads, head_dim)
        ctx_tensor = ctx_tensor.transpose(1, 2)
        ctx_tensor = ctx_tensor.contiguous().view(b, seq_len, self.d_out)
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

    d_in = inputs.shape[-1]
    d_out = 12
    multi_head = GroupedQueryAttention(
        d_in=d_in,
        head_dim=2,
        num_heads=6,
        num_kv_groups=2,
    )
    print(multi_head(input_batch, mask, cos, sin))
