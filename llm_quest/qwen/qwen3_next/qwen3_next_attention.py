import torch
import torch.nn as nn

from llm_quest.common.rope import RoPE

# Differences compared to `Qwen3_attention.py`:
#
# We now have a hybrid attention architecture alternating between:
#
# - A modified classic quadratic attention block:
#       We removed all QK Norm and use Zero-Centered RMSNorm instead
#       The classic Attention block is now a Gated SDPA block (basically GQA with an extra sigmoid activated gate)
#       RoPE is only applied to the first 25% of the head dimensions
#       GQA was implemented multiple times in the repo, for a change, using Pytorch's SDPA function
#
# - A new subquadratic attention block: Gated Delta Net


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
        return (x * rms * (1.0 + self.scale)).to(input_dtype)  # fullcast to fp32 before returning to input dtype


def l2_norm(x):
    """
    Reducing Q and K vectors magnitude to unit length, by dividing by their L2 norms.
    Ie, we focus on the direction of the vectors, by making the similarity measure invariant to their magnitudes.
    This is only done for GatedDeltaNet.

    args: x: (b, num_head, seq_len, head_dim) should be Q or K
    """
    l2_norm = torch.linalg.vector_norm(x, dim=-1, ord=2, keepdim=True)
    return x * torch.clamp(l2_norm, min=1e-6).reciprocal()


# NOTE This was made as a separate helper function because it really needed some more explanation
def compute_alpha_factor(log_A, a, dt_bias):
    """
    Calculates the state decay factor alpha following Qwen3-Next/SSM-style formula.

    Alpha is the exponential decay factor applied to the previous state memory in Gated Delta Rule.
    This controls how much of the previous state memory we keep or forget.

    alpha = e^(-A * Δt) (can be seen as e^(-Rate * Time)) where A > 0 and Δt > 0:
    - A is learned as log_A and then exponentiated (e^log_A) to ensure positivity.
    - Δt is passed through a softplus to ensure positivity.
    The positivity of both terms ensure that alpha, via the neg exponent e^-, is always in (0, 1) as a final decay
    factor.

    Δt is the result of the affine function Wx + dt_bias with "a" as Wx (this makes Δt dynamic per token and thus the
    decay)
    Δt represents how much duration to apply the decay (time step).

    args:
        log_A: (num_v_heads,) represents the base (log) decay rate per value head (will be a constant per head)
        a: (b, seq_len, num_v_heads) the tokens to num_v_heads projections (will be dynamic per token)
        dt_bias: (num_v_heads,) learnable bias for time step Δt

    returns:
        alpha: (b, seq_len, num_v_heads) final decay factor per token, range (0, 1)
    """
    A = torch.exp(log_A)  # retrieves positive A from the learned logarithm
    delta_t = torch.nn.functional.softplus(a + dt_bias)  # Δt

    alpha = torch.exp(-A * delta_t)  # e^(-Rate * Time)
    return alpha


def gated_delta_rule(queries, keys, values, beta, alpha, prev_state=None):
    """
    Gated Delta Rule following equation 10 from the paper: GATED DELTA NETWORKS: IMPROVING MAMBA2 WITH DELTA RULE
    which is slightly different in terms of calculation than Qwen3-Next (they are doing transposed S_t^T)

    More details:
    https://github.com/casinca/LLM-quest/blob/master/llm_quest/qwen/qwen3_next/README.md#making-sense-of-the-gated-delta-rule-equation-and-the-code

    args:
        queries: (b, num_heads, seq_len, qk_head_dim)
        keys: (b, num_heads, seq_len, qk_head_dim)
        values: (b, num_heads, seq_len, v_head_dim)
        beta: (b, num_v_heads, seq_len) writing strength/learning rate per value head and per token
        alpha: (b, num_v_heads, seq_len) state decay factor: how much of previous state/memory we keep ϵ (0, 1)
            this is a per head and per token scalar factor (same as beta)
            -if ~0 forget almost everything
            -if ~1 remember almost everything
        prev_state: (b, num_heads, v_head_dim, k_head_dim) previous state/memory.

    returns:
        attn_output: (b, num_heads, seq_len, v_head_dim) the context tensor
        prev_state: updated state/memory to be used in the next forward pass
    """
    # NOTE: we previously interleaved Q and K to V, thus now num_heads = num_qk_heads = num_vg_heads
    b, num_heads, seq_len, k_head_dim = keys.shape
    v_head_dim = values.shape[-1]
    scale = queries.shape[-1] ** -0.5  # scaling factor for queries (same as attention scaling)

    # performing the calculation in fp32
    initial_dtype = queries.dtype
    queries, keys, values, beta, alpha = map(lambda t: t.to(torch.float32), (queries, keys, values, beta, alpha))
    queries *= scale

    attn_output = torch.zeros_like(values)
    if prev_state is None:  # same shape/outer product as the paper ie S_t = vk^T (Qwen is doing kv^T = S_t^T))
        prev_state = torch.zeros(b, num_heads, v_head_dim, k_head_dim, dtype=values.dtype, device=values.device)

    for t in range(seq_len):
        k_t = keys[:, :, t, :]  # (b, num_heads, k_head_dim)
        v_t = values[:, :, t, :]
        q_t = queries[:, :, t, :]
        beta_t = beta[:, :, t].unsqueeze(-1)  # (b, num_v_heads, 1)
        alpha_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (b, num_v_heads, 1, 1)

        # applying, per head for that token the decay factor to previous state (scalar product)
        gated_prev_state = alpha_t * prev_state  # (b, num_heads, v_head_dim, k_head_dim)
        # retrieving old value associated to current key, in order to calc the error/delta (doing vector matrix product)
        v_old = gated_prev_state @ k_t.unsqueeze(-1)  # (b, num_heads, v_head_dim, 1)
        delta = v_t - v_old.squeeze(-1)  # (b, num_heads, v_head_dim)
        scaled_delta = beta_t * delta  # scalar product
        state_update = scaled_delta.unsqueeze(-1) @ k_t.unsqueeze(2)  # (b, num_heads, v_head_dim, k_head_dim)
        prev_state = gated_prev_state + state_update  # now is S_t

        attn_t = prev_state @ q_t.unsqueeze(-1)  # (b, num_heads, v_head_dim, 1)
        attn_output[:, :, t, :] = attn_t.squeeze(-1)

    return attn_output.to(initial_dtype), prev_state


class GatedAttention(nn.Module):
    """
    Gated Scaled Dot Product Attention(SDPA) using GQA as described in Qwen3-Next blogpost.
    It's similar to `Qwen3_attention.py` but with an added sigmoid activated gate and Zero-Centered RMSNorm for QK
    normalization

    Using Pytorch's built-in SDPA function for Attention calc

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.d_in = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.num_kv_groups = cfg["num_kv_groups"]
        assert self.num_heads % self.num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        self.head_dim = cfg["head_dim"]
        self.d_out = self.num_heads * self.head_dim
        self.dtype = cfg["dtype"]

        self.num_repeat = self.num_heads // self.num_kv_groups
        self.p_dropout = cfg["p_dropout"] if cfg["training"] else 0.0

        self.w_queries = nn.Linear(self.d_in, self.d_out, bias=False, dtype=self.dtype)
        self.w_keys = nn.Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=False, dtype=self.dtype)
        self.w_values = nn.Linear(self.d_in, self.num_kv_groups * self.head_dim, bias=False, dtype=self.dtype)
        self.w_gate = nn.Linear(self.d_in, self.d_out, bias=False, dtype=self.dtype)

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, dtype=self.dtype)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, dtype=self.dtype)

        self.out_proj = nn.Linear(self.d_out, self.d_in, bias=False, dtype=self.dtype)

    def forward(self, x, mask, cos, sin, attn_mask=None):
        """
        args:
            x: (b, seq_len, d_in)
            mask: (seq_len, seq_len) causal mask, should come inverted from GlobalBuffers (for SDPA function)
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
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


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Net as described in Qwen3-Next blogpost "diagram".
    This approach is less efficient because linear and conv layers are not fused but is easier to follow along as a from
    scratch implementation.
    We are also doing the recurrent Gated Delta Rule and not the optimized Chunked version from FLA/parallel GDN paper.

    It is true that fusing in GDN is even more incentivized since we have 6 linear transformations with: Q,K,V,G
    and alpha, beta. And speed is the highlight of Qwen3-Next arch.
    """

    def __init__(self, cfg):
        super().__init__()

        self.d_in = cfg["emb_dim"]
        self.num_qk_heads = cfg["linear_num_qk_heads"]
        self.num_v_heads = cfg["linear_num_value_heads"]
        self.qk_head_dim = cfg["linear_qk_head_dim"]  # same for Q and K
        self.vg_head_dim = cfg["linear_value_head_dim"]  # same for V and gate
        self.conv_kernel_size = cfg["linear_conv_kernel_size"]
        self.num_repeat = self.num_v_heads // self.num_qk_heads  # similar to GQA (here "GV" grouped value heads)

        self.d_out = self.num_qk_heads * self.qk_head_dim
        self.d_out_vg = self.num_v_heads * self.vg_head_dim

        self.dtype = cfg["dtype"]
        self.p_dropout = cfg["p_dropout"] if cfg["training"] else 0.0

        self.w_queries = nn.Linear(self.d_in, self.d_out, bias=False, dtype=self.dtype)
        self.w_keys = nn.Linear(self.d_in, self.d_out, bias=False, dtype=self.dtype)
        self.w_values = nn.Linear(self.d_in, self.d_out_vg, bias=False, dtype=self.dtype)

        # projection to num_v_heads: this what enables dynamicity of the factors (ie per token) for each value head
        self.w_beta = nn.Linear(self.d_in, self.num_v_heads, bias=False, dtype=self.dtype)
        self.w_alpha = nn.Linear(self.d_in, self.num_v_heads, bias=False, dtype=self.dtype)

        # alpha components, to calc alpha decay factor following Qwen3-Next here, see compute_alpha_factor()
        A_init = torch.empty(self.num_v_heads, dtype=self.dtype).uniform_(0, 16)
        self.log_A = nn.Parameter(torch.log(A_init))  # log_A to ensure A > 0
        self.dt = nn.Parameter(torch.ones(self.num_v_heads, dtype=self.dtype))

        self.activation = nn.SiLU()
        self.post_norm = ZeroCenteredRMSNorm(self.d_out_vg, dtype=self.dtype)
        self.w_gate = nn.Linear(self.d_in, self.d_out_vg, bias=False, dtype=self.dtype)
        self.out_proj = nn.Linear(self.d_out_vg, self.d_in, bias=False, dtype=self.dtype)

        self.conv_queries = nn.Conv1d(
            in_channels=self.d_out,
            out_channels=self.d_out,  # number of kernels/filters
            kernel_size=self.conv_kernel_size,
            bias=False,
            padding=self.conv_kernel_size - 1,  # serves as causal mask
            dtype=self.dtype,
        )
        self.conv_keys = nn.Conv1d(
            in_channels=self.d_out,
            out_channels=self.d_out,
            kernel_size=self.conv_kernel_size,
            bias=False,
            padding=self.conv_kernel_size - 1,
            dtype=self.dtype,
        )
        self.conv_values = nn.Conv1d(
            in_channels=self.d_out_vg,
            out_channels=self.d_out_vg,
            kernel_size=self.conv_kernel_size,
            bias=False,
            padding=self.conv_kernel_size - 1,
            dtype=self.dtype,
        )

    def forward(self, x, attn_mask=None):
        """
        args:
            x: (b, seq_len, d_in)
            attn_mask (optional): (b, seq_len) used for padding tokens, from collators 1=real token, 0=padding
        """
        b, seq_len, d_in = x.shape

        # We mask at the beginning (vs classic attention) because of the conv layers
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, seq_len, 1)
            x *= attn_mask

        # shape (b, seq_len, d_out) → (b, d_out, seq_len) for Conv1D expecting that shape
        queries = self.w_queries(x).transpose(1, 2)
        keys = self.w_keys(x).transpose(1, 2)
        values = self.w_values(x).transpose(1, 2)

        # We are not doing features convolution but a temporal convolution, (ie over the sequence length)
        queries = self.activation(self.conv_queries(queries))[..., :seq_len]
        keys = self.activation(self.conv_keys(keys))[..., :seq_len]
        values = self.activation(self.conv_values(values))[..., :seq_len]

        # reshaping to multiheads for attention: (b, d_out, seq_len) → (b, num_heads, seq_len, head_dim)
        queries = queries.reshape(b, self.num_qk_heads, self.qk_head_dim, -1).transpose(2, 3).contiguous()
        keys = keys.reshape(b, self.num_qk_heads, self.qk_head_dim, -1).transpose(2, 3).contiguous()
        values = values.reshape(b, self.num_v_heads, self.vg_head_dim, -1).transpose(2, 3).contiguous()

        queries = l2_norm(queries)
        keys = l2_norm(keys)

        if self.num_repeat > 1:  # per the current Qwen3-Next config this should always be the case
            queries = queries.repeat_interleave(self.num_repeat, dim=1)
            keys = keys.repeat_interleave(self.num_repeat, dim=1)

        # shape (b, num_v_heads, seq_len) for beta and alpha
        beta = torch.sigmoid(self.w_beta(x).transpose(1, 2).contiguous())
        token_projs = self.w_alpha(x)
        alpha = compute_alpha_factor(self.log_A, token_projs, self.dt).transpose(1, 2).contiguous()

        # state isn't needed for training unlike inference
        ctx_tensor, prev_state = gated_delta_rule(queries, keys, values, beta, alpha, prev_state=None)

        # reshaping (b, num_head, seq_len, v_head_dim) → (b, seq_len, d_out_vg) for the gate scaling
        ctx_tensor = ctx_tensor.transpose(1, 2).contiguous().view(b, seq_len, self.d_out_vg)
        ctx_tensor = self.post_norm(ctx_tensor)

        gate_output = self.activation(self.w_gate(x))  # shape (b, seq_len, d_out_vg)
        output = gate_output * ctx_tensor

        output = self.out_proj(output)  # shape (b, seq_len, d_in)

        return output


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

    input_batch = torch.stack((inputs, inputs), dim=0).bfloat16()
    attn_mask = torch.tensor([[1, 1, 1, 1, 0, 0]]).repeat(2, 1)

    # context length/ seq length (b, s, emb_dim)
    ctx_len = input_batch.shape[1]

    mask, cos, sin = GlobalBuffers().get_buffers(ctx_len, 10_000, 2)
    mask = ~mask  # for backward compatibility inverting (SPDA:Mask where False OUR:Mask where True)

    dummy_cfg = {
        "emb_dim": inputs.shape[-1],
        "n_heads": 6,
        "head_dim": 2,
        "num_kv_groups": 2,
        "linear_num_qk_heads": 2,
        "linear_num_value_heads": 4,
        "linear_qk_head_dim": 2,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_size": 3,
        "p_dropout": 0.0,
        "training": True,
        "dtype": torch.bfloat16,
    }

    with torch.no_grad():
        gsdpa = GatedAttention(dummy_cfg)
        gdnet = GatedDeltaNet(dummy_cfg)

    print(gsdpa(input_batch, mask, cos, sin))
    print(f"\n{'*'*50}\n")
    print(gdnet(input_batch, attn_mask))  # last 2 vectors masked per attention mask
