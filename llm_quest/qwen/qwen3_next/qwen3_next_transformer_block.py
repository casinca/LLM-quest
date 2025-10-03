import torch.nn as nn

from llm_quest.moe.qwen3_moe import Qwen3MoE
from llm_quest.qwen.qwen3_next.qwen3_next_attention import GatedAttention, GatedDeltaNet, ZeroCenteredRMSNorm


class Qwen3NextTransformerBlock(nn.Module):
    """
    Qwen3-Next transformer block

    Differences from Qwen3:

    The layer idx is used to determine which attention module to use:
    - GatedAttention: every cfg["linear_sdpa_ratio"] layers
    - GatedDeltaNet: every other layer

    - Zero-Centered RMSNorm is replacing RMSNorm
    - MoE also include a weighted shared expert

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
        layer_idx (int): Layer index
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        # hybrid attention architecture: alternating between GatedDeltaNet and GatedAttention
        ratio = cfg["linear_sdpa_ratio"]
        self.att = GatedDeltaNet(cfg) if (layer_idx + 1) % ratio else GatedAttention(cfg)
        self.norm1 = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm2 = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.moe = Qwen3MoE(cfg=cfg)

    def forward(self, x, mask, cos, sin):

        residual = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.moe(x)
        x = x + residual

        return x
