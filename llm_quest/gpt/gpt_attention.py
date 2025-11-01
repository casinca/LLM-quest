from math import sqrt

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.w_queries = nn.Parameter(torch.rand(d_in, d_out))
        self.w_keys = nn.Parameter(torch.rand(d_in, d_out))
        self.w_values = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.w_queries
        keys = x @ self.w_keys
        values = x @ self.w_values

        d_k = sqrt(keys.shape[-1])  # scaler, square root of the embeddings keys (so last dim)

        raw_att = queries @ keys.T
        scaled_raw_att = raw_att / d_k
        norm_att = torch.softmax(scaled_raw_att, dim=-1)
        ctx_tensor = norm_att @ values

        return ctx_tensor


# optimized using 1xFFN Linear layer (ax+b) and no biases, which ends up being a*x. The Linear classes also help with
# controlling the output dim (reducing/compression if d_out > d_in or expansion d_out > d_int, GPT has d_in = d_out)
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        d_k = sqrt(keys.shape[-1])  # scaler, square root of the embeddings keys (so last dim)

        raw_att = queries @ keys.T
        scaled_raw_att = raw_att / d_k
        norm_att = torch.softmax(scaled_raw_att, dim=-1)
        ctx_tensor = norm_att @ values

        return ctx_tensor


# optimized for decoders causal attention, to keep the autoregressive property, we mask future tokens, so that we only
# attend to self and previous ones. Also adding dropout for regularization
class SelfAttention_v3(nn.Module):
    def __init__(self, d_in, d_out, dropout, ctx_len, qkv_bias=False):
        super().__init__()
        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # registering mask tensor (buffers, != nn.Parameter, are non-learnable params and moves with the model too)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1))

    def forward(self, x):
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        d_k = sqrt(keys.shape[-1])  # scaler, square root of the num of embeddings keys (so last dim)

        raw_att = queries @ keys.mT  # .mT transpose last 2 dims (alt to transpose(1,2))
        scaled_raw_att = raw_att / d_k
        # masking_ (in place and not reassigning to a new var for memory)
        # Slicing dynamically to actual seq len and not context len (in case of inputs of variable length)
        sequ_len = x.shape[1]
        scaled_raw_att.masked_fill_(self.mask.bool()[:sequ_len, :sequ_len], -torch.inf)
        norm_att = torch.softmax(scaled_raw_att, dim=-1)

        norm_att = self.dropout(norm_att)  # regularization
        ctx_tensor = norm_att @ values

        return ctx_tensor


# MultiHead
# unoptimized way: looping multiple instances of SelfAttention class (so sequentially) and concatenate their
# contextual scalars to form a vector for each word, of shape d_out * num_heads
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, dropout, ctx_len, num_heads, qkv_bias=False):
        super().__init__()
        self.multi_context = nn.ModuleList(
            [SelfAttention_v3(d_in, d_out, dropout, ctx_len, qkv_bias) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out * num_heads, d_out * num_heads)

    def forward(self, x):
        multi_ctx_concat = torch.concat([head(x) for head in self.multi_context], dim=-1)

        return self.out_proj(multi_ctx_concat)


# Old KVcache ref: https://github.com/casinca/LLM-quest/commit/0cbf60a078560e77e96cd0ef9a16803f7e5e3240
class KVCache:
    """
    KV cache with preallocation and updating by directly indexing into the cache.
    We avoid torch.cat() operations from the old KVcache.

    Args:
        num_layers (int): Number of transformer layers
        context_len (int): Maximum context_length (max sequence length)
    """

    def __init__(self, num_layers, context_len):
        self.num_layers = num_layers
        self.context_len = context_len

        self.keys_cache = None
        self.values_cache = None

        self.start_pos = 0  # track current sequence length

    def _initialize(self, batch_size, num_heads, head_dim, device, dtype):
        """
        initialize cache tensors on first call
        """

        self.keys_cache = []
        self.values_cache = []

        for _ in range(self.num_layers):
            self.keys_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    self.context_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.values_cache.append(
                torch.zeros(
                    batch_size,
                    num_heads,
                    self.context_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )

    def get_updated_cache(self, keys, values, layer_idx):
        """
        Update cache with the new keys and values and return the full cached keys and values.

        Args:
            keys: New keys tensor of shape (batch_size, num_heads, new_seq_len, head_dim)
            values: New values tensor of shape (batch_size, num_heads, new_seq_len, head_dim)
            layer_idx: Layer index to update

        Note: In most scenarios new_seq_len should be 1

        Returns:
            A tuple containing the full cached keys and values up to the current sequence length.
        """
        batch_size, num_heads, new_seq_len, head_dim = keys.shape

        if self.keys_cache is None and self.values_cache is None:
            self._initialize(batch_size, num_heads, head_dim, keys.device, keys.dtype)

        end_pos = self.start_pos + new_seq_len

        # update from the new keys and values into the pre-allocated cache
        self.keys_cache[layer_idx][:, :, self.start_pos : end_pos, :] = keys
        self.values_cache[layer_idx][:, :, self.start_pos : end_pos, :] = values

        # update sequence length after the last layer has been processed for the next call
        if layer_idx == self.num_layers - 1:
            self.start_pos += new_seq_len

        # return slices of the cache, up to the new current sequence length
        return (
            self.keys_cache[layer_idx][:, :, :end_pos, :],
            self.values_cache[layer_idx][:, :, :end_pos, :],
        )


# MHA optimized, splitting our tensors per head then merging back
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module that processes input through multiple attention heads in parallel.

    This implementation optimizes the attention computation by splitting tensors across heads
    rather than using separate attention modules. The input embeddings are projected to queries,
    keys and values, split across heads, and processed in parallel before being concatenated.

    Args:
        d_in (int): Input embedding dimension
        d_out (int): Output embedding dimension (must be divisible by num_heads)
        dropout (float): Dropout probability
        ctx_len (int): Maximum context/sequence length for positional masking
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Whether to include bias terms in query/key/value projections.
                                Defaults to False.
        layer_idx (int): Layer index (used here for the KV cache indexing)

    Attributes:
        head_dim (int): Dimension of each attention head (d_out // num_heads)
        w_queries (nn.Linear): Query projection layer
        w_keys (nn.Linear): Key projection layer
        w_values (nn.Linear): Value projection layer
        dropout (nn.Dropout): Dropout layer
        mask (torch.Tensor): Attention mask for causal attention
        out_proj (nn.Linear): Output projection layer
    """

    def __init__(self, d_in, d_out, dropout, ctx_len, num_heads, qkv_bias=False, layer_idx=None):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool())
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads
        self.att_scaling = self.head_dim**-0.5
        self.out_proj = nn.Linear(d_out, d_out)  # optional additional learnable params for the output
        self.layer_idx = layer_idx

    def forward(self, x, attn_mask=None, kv_cache=None):
        """
        args:
            x: (b, seq_len, d_in)
            attn_mask (optional): (b, seq_len) used for padding tokens, passed as True = real tokens
            kv_cache (optional): KVCache instance/object
        """
        queries = self.w_queries(x)  # shape (b, s, d_out) aka augmented emb dim if d_out > d_in
        keys = self.w_keys(x)
        values = self.w_values(x)

        b, seq_len, d_in = x.shape
        # reshaping/splitting at the d_out dim our 3D tensors by the num of heads, reshaped, is a 4D tensor:
        # (batch, seq_len, num of heads, head emb dim) ex: 2,6,4 → (2heads) → 2,6,2,2
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_heads, -1)
        values = values.view(b, seq_len, self.num_heads, -1)

        # transposing first, head_dim and seq len for correct matmul within each head, ex 2,6,2,2 → 2,2,6,2
        queries = torch.transpose(queries, 1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # --- KV Cache update ---
        if kv_cache is not None:
            keys, values = kv_cache.get_updated_cache(keys, values, self.layer_idx)

        att_scores = queries @ keys.mT  # shape (b, num_heads, seq_len, seq_len) or w/ KVc (b, num_heads, 1, seq_len)
        scaled_att_scores = att_scores * self.att_scaling

        # --- Masking ---
        # with KVcache, Q and K don't have the same length anymore, can't just use "seq_len" but for training will work
        # too, since q_seq_len = k_seq_len = seq_len
        q_seq_len = queries.shape[2]
        k_seq_len = keys.shape[2]
        # mask fix for KV cache (sequence length Q and K mismatch)
        if k_seq_len > q_seq_len:
            q_start_pos = k_seq_len - q_seq_len  # should be 1 for classic NTP KVCache inference
            curr_mask = self.mask[q_start_pos:k_seq_len, :k_seq_len]
        else:
            curr_mask = self.mask[:q_seq_len, :k_seq_len]

        if attn_mask is not None:
            # reshape & combine masks (invert attn_mask to get True = padding)
            curr_mask = curr_mask.view(1, 1, q_seq_len, k_seq_len) | ~attn_mask.view(b, 1, 1, k_seq_len)
        # using a small value instead of -inf for padding tokens attending to padding tokens:
        # this is an edge case in left padding where pad x pad becomes a full vector of -infs and softmax will NaN.
        # https://github.com/huggingface/transformers/issues/32390
        mask_value = torch.finfo(scaled_att_scores.dtype).min / 2
        scaled_att_scores.masked_fill_(curr_mask, mask_value)  # mask where True

        att_weights = torch.softmax(scaled_att_scores, dim=-1)
        att_weights = self.dropout(att_weights)  # reg

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


# testing code
if __name__ == "__main__":
    torch.manual_seed(123)
    # EX WEIGHTS AND ATTENTION
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your (x^1)
            [0.55, 0.87, 0.66],  # journey (x^2)
            [0.57, 0.85, 0.64],  # starts (x^3)
            [0.22, 0.58, 0.33],  # with (x^4)
            [0.77, 0.25, 0.10],  # one (x^5)
            [0.05, 0.80, 0.55],  # step (x^6)
        ]
    )
    attn_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    d_in = inputs.shape[-1]
    d_out = 2

    # V2 -------------------

    attv2 = SelfAttention_v2(d_in, d_out)

    key_t = attv2.w_keys(inputs)
    query_t = attv2.w_queries(inputs)

    raw_att = query_t @ key_t.T
    # print(raw_att)

    mask = torch.triu(torch.ones(raw_att.shape), diagonal=1)
    masked_raw_att = raw_att.masked_fill(mask.bool(), -torch.inf)
    # print(masked_raw_att)

    # alt
    # mask_raw_att = torch.tril(raw_att, diagonal=0)
    # mask_raw_att = mask_raw_att.masked_fill(mask_raw_att == 0, float("-inf"))
    # print(mask_raw_att)

    # V3 -------------------

    input_batch = torch.stack((inputs, inputs), dim=0)
    attn_mask = torch.stack((attn_mask, attn_mask), dim=0)
    # context length/ seq length (b, s, emb_dim)
    ctx_len = input_batch.shape[1]

    attv3 = SelfAttention_v3(d_in, d_out, 0.1, ctx_len)
    # print(attv3.forward(input_batch))

    # MHAs -------------------

    # print("unoptim_multi_head\n")
    unoptim_multi_head = MultiHeadAttentionWrapper(d_in, d_out, 0.5, 6, 2)
    # print(unoptim_multi_head.forward(input_batch))

    print("multi_head\n")
    multi_head = MultiHeadAttention(d_in, d_out, 0.0, 6, 2)
    print(multi_head(input_batch, attn_mask))
