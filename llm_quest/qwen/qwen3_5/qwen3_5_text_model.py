import torch
import torch.nn as nn

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.qwen.qwen3.qwen3_transformer_block import FFN
from llm_quest.qwen.qwen3_next.qwen3_next_attention import (
    GatedAttention,
    ZeroCenteredRMSNorm,
    compute_alpha_factor,
    gated_delta_rule,
    l2_norm,
)

# The full_attention layers (`GatedAttention` class) are re-used from Qwen3-Next in `qwen3_next_attention.py`.
# So nothing changes for these.
#
# However GDN needs some tweaks, differences compared to `qwen3_next_attention.py`:
#
# The linear_attention layers ( in `GatedDeltaNet` class) are modified to use fused weights:
# - QKV are projected with a single fused linear `w_qkv` instead of 3 separate linears
#
# - A single Conv1d operates on the fused QKV output instead of 3 separate Conv1ds
#
# - The gate is kept as a separate projection `w_gate` (was fused in Qwen3-Next official implementation)
#
# - Beta and Alpha are kept as separate projections `w_beta` and `w_alpha` (just like in `GatedDeltaNet` class)
#   but in the official Qwen3-Next, these were fused. So Qwen reverted to separate projections for Qwen3.5.
#
# These fusing changes are now needed because we want to match the pretrained weight structure from HF
# for weight loading.


class FusedGatedDeltaNet(nn.Module):
    """
    Fused Gated DeltaNet for Qwen3.5.

    Same as our Qwen3-Next `GatedDeltaNet` class but with some fused weights:
    - QKV are projected in a single fused linear (w_qkv) instead of 3 separate ones
    - A single Conv1d is used on the fused QKV output instead of 3 separate Conv1ds

    Rest is the same, we re-use: l2_norm(), compute_alpha_factor(), gated_delta_rule() from `qwen3_next_attention.py`
    """

    def __init__(self, cfg):
        super().__init__()

        self.d_in = cfg["emb_dim"]
        self.num_qk_heads = cfg["linear_num_qk_heads"]
        self.num_v_heads = cfg["linear_num_value_heads"]
        self.qk_head_dim = cfg["linear_qk_head_dim"]
        self.vg_head_dim = cfg["linear_value_head_dim"]
        self.conv_kernel_size = cfg["linear_conv_kernel_size"]
        self.num_repeat = self.num_v_heads // self.num_qk_heads

        self.d_out = self.num_qk_heads * self.qk_head_dim  # dim for Q and K
        self.d_out_vg = self.num_v_heads * self.vg_head_dim  # dim for V and gate

        self.dtype = cfg["dtype"]

        # Fused QKV projection, single linear
        self.w_qkv = nn.Linear(self.d_in, self.d_out * 2 + self.d_out_vg, bias=False, dtype=self.dtype)
        self.w_gate = nn.Linear(self.d_in, self.d_out_vg, bias=False, dtype=self.dtype)
        self.w_beta = nn.Linear(self.d_in, self.num_v_heads, bias=False, dtype=self.dtype)
        self.w_alpha = nn.Linear(self.d_in, self.num_v_heads, bias=False, dtype=self.dtype)

        # Alpha decay components (same as Qwen3-Next) Important: A_log is in fp32
        A_init = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.log_A = nn.Parameter(torch.log(A_init))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=self.dtype))

        # Single conv1d on fused QKV (replaces our 3 separate convs in Qwen3-Next GDN)
        # the goal is to inject local positional context along the time axis (temporal/depthwise causal convolution),
        # not to mix features across dimensions (standard convolution)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_out * 2 + self.d_out_vg,
            out_channels=self.d_out * 2 + self.d_out_vg,  # number of kernels/filters
            kernel_size=self.conv_kernel_size,
            bias=False,
            padding=self.conv_kernel_size - 1,  # serves as causal mask (causal convolution)
            # depthwise conv/per channel independently (groups=channels), not mixed/standard conv (groups=1)
            groups=self.d_out * 2 + self.d_out_vg,
            dtype=self.dtype,
        )

        self.activation = nn.SiLU()
        self.post_norm = PytorchRMSNorm(self.vg_head_dim, dtype=torch.float32)  # in fp32, being explicit about it
        self.out_proj = nn.Linear(self.d_out_vg, self.d_in, bias=False, dtype=self.dtype)

    def forward(self, x, attn_mask=None):
        """
        args:
            x: (b, seq_len, d_in)
            attn_mask (optional): (b, seq_len) used for padding tokens, from collators 1=real token, 0=padding
        """
        b, seq_len, d_in = x.shape

        # Mask padding tokens at the beginning for the conv layer (same as Qwen3-Next GDN)
        if attn_mask is not None:
            x *= attn_mask.view(b, seq_len, 1)

        # Fused QKV projection
        fused_qkv = self.w_qkv(x)  # (b, seq_len, d_out*2 + d_out_vg=fused_dim)
        # shape (b, num_v_heads, seq_len) for beta and alpha
        beta = torch.sigmoid(self.w_beta(x).transpose(1, 2).contiguous())
        token_projs = self.w_alpha(x)
        alpha = compute_alpha_factor(self.log_A, token_projs, self.dt_bias).transpose(1, 2).contiguous()

        # Temporal convolution on fused QKV (ie over the sequence length)
        # shape (b, seq_len, d_out) → (b, d_out, seq_len) for Conv1D expecting that shape
        fused_qkv = fused_qkv.transpose(1, 2)
        fused_qkv = self.activation(self.conv1d(fused_qkv)[..., :seq_len])
        fused_qkv = fused_qkv.transpose(1, 2)  # back to (b, seq_len, fused_dim)

        # Split back into Q, K, V after convolution
        query, key, value = torch.split(
            fused_qkv,
            [self.d_out, self.d_out, self.d_out_vg],
            dim=-1,
        )

        # reshape+transpose to multiheads for attention: (b, d_out, seq_len) → (b, num_heads, seq_len, head_dim)
        query = query.reshape(b, seq_len, self.num_qk_heads, self.qk_head_dim).transpose(1, 2).contiguous()
        key = key.reshape(b, seq_len, self.num_qk_heads, self.qk_head_dim).transpose(1, 2).contiguous()
        value = value.reshape(b, seq_len, self.num_v_heads, self.vg_head_dim).transpose(1, 2).contiguous()

        query = l2_norm(query)
        key = l2_norm(key)

        # Interleave Q/K heads to match V heads (same as Qwen3-Next)
        if self.num_repeat > 1:
            query = query.repeat_interleave(self.num_repeat, dim=1)
            key = key.repeat_interleave(self.num_repeat, dim=1)

        ctx_tensor, prev_state = gated_delta_rule(query, key, value, beta, alpha, prev_state=None)

        # The norm+gate is done in fp32
        ctx_tensor = self.post_norm(ctx_tensor.to(torch.float32))
        # reshaping (b, num_head, seq_len, v_head_dim) → (b, seq_len, d_out_vg) for the gate scaling
        ctx_tensor = ctx_tensor.transpose(1, 2).contiguous().view(b, seq_len, self.d_out_vg)

        gate_output = self.activation(self.w_gate(x).to(torch.float32))  # (b, seq_len, d_out_vg)
        output = (gate_output * ctx_tensor).to(self.dtype)

        output = self.out_proj(output)  # shape (b, seq_len, d_in)

        return output


class Qwen3_5TransformerBlock(nn.Module):
    """
    Qwen3.5 transformer block with hybrid attention architecture.

    Differences from Qwen3-Next (qwen3_next_transformer_block.py):
    - Uses FusedGatedDeltaNet for linear_attention layers (fused to match HF pretrained weights)
    - Dense SwiGLU FFN (from Qwen3) instead of MoE for loading smaller models

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
        layer_idx (int): Layer index (0-based) to determine which attention module to use
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()

        interval = cfg["linear_sdpa_ratio"]
        # hybrid attention architecture: alternating between FusedGatedDeltaNet and GatedAttention
        self.att = FusedGatedDeltaNet(cfg) if (layer_idx + 1) % interval else GatedAttention(cfg)
        self.norm1 = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm2 = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.ffn = FFN(cfg)

    def forward(self, x, mask, cos, sin, attn_mask=None):
        """
        args:
            x: (b, seq_len, emb_dim)
            mask: (seq_len, seq_len) causal mask (inverted for SDPA: True = masked)
            cos, sin: RoPE cos/sin
            attn_mask: (b, seq_len) 1=real token, 0=padding
        """
        residual = x
        x = self.norm1(x)

        # dispatching based on attention type (full/classic attention or linear attention)
        x = self.att(x, mask, cos, sin, attn_mask) if isinstance(self.att, GatedAttention) else self.att(x, attn_mask)

        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


# Differences from Qwen3-Next model:
#
# - Uses dense SwiGLU FFN instead of MoE (handled by the transformer block)
# - Uses FusedGatedDeltaNet (fused weights) for linear_attention layers
# - TODO No MRoPE yet later with vision, text generation checking first
class Qwen3_5Model(nn.Module):
    """
    Qwen3.5 implementation, similar to Qwen3-Next at this level of the architecture:
    - We pass the layer idx to the transformer block to determine which attention block to use
    - use Zero-Centered RMSNorm

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.tie_embeddings = cfg["tie_embeddings"]

        self.emb_dict = nn.Embedding(
            num_embeddings=cfg["vocab_size"],
            embedding_dim=cfg["emb_dim"],
            dtype=cfg["dtype"],
        )

        self.trf_blocks = nn.ModuleList(
            [Qwen3_5TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["n_layers"])]
        )

        self.final_norm = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])

        # Weight tying based on model configuration
        # this part is only useful for either: pretraining or reducing memory allocation before loading weights
        if self.tie_embeddings:
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"], device="meta")
            assert self.tie_embeddings and self.emb_dict.weight.shape == self.out_head.weight.shape, (
                "Shape mismatch for weight tying"
            )
            self.out_head.weight = self.emb_dict.weight
            nn.init.xavier_uniform_(self.out_head.weight)
        else:
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Initialize RoPE and causal mask buffers
        mask = GlobalBuffers.get_causal_mask(cfg["context_length"])
        cos, sin = GlobalBuffers.get_rope_params(
            ctx_len=cfg["context_length"],
            rope_base=cfg["rope_base"],
            head_dim=cfg["head_dim"],
            rotation_factor=cfg["partial_rope_factor"],
        )
        self.register_buffer("mask", ~mask)  # Inverted for compatibility with Pytorch's SDPA function
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, attn_mask=None):
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin, attn_mask)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


# quick inline tests
if __name__ == "__main__":
    import config

    torch.manual_seed(123)

    # Test 1: FusedGatedDeltaNet
    print("--- Testing FusedGatedDeltaNet ---")

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
        "training": False,
        "dtype": torch.bfloat16,
    }

    with torch.no_grad():
        fused_gdn = FusedGatedDeltaNet(dummy_cfg)

    print("FusedGatedDeltaNet output:")
    print(fused_gdn(input_batch, attn_mask))  # last 2 vectors masked per attention mask
    print("-" * 40)

    # Test 2: Qwen3_5Model
    print("\n--- Testing Qwen3_5Model ---")

    torch.manual_seed(123)
    model = Qwen3_5Model(config.QWEN3_5_08B_CONFIG)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    sample_input = torch.randint(0, 1000, (2, 10))  # b=2, seq_len=10
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0]}")
