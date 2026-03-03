import torch.nn as nn

from llm_quest.qwen.qwen3.qwen3_transformer_block import FFN
from llm_quest.qwen.qwen3_5.qwen3_5_attention import FusedGatedDeltaNet
from llm_quest.qwen.qwen3_next.qwen3_next_attention import GatedAttention, ZeroCenteredRMSNorm


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
