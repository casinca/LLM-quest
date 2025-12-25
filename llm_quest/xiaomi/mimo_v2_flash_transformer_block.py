import torch.nn as nn

from llm_quest.moe.deepseek_moe import DeepSeekMoE
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.xiaomi.mimo_v2_flash_attention import GroupedQueryAttentionWithSink


class FFN(nn.Module):
    """FFN reused from Qwen3."""

    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.lin_gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.silu_activ = nn.functional.silu
        self.lin2 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_gate = self.lin_gate(x)
        x_gate = self.silu_activ(x_gate)
        x1 = self.lin1(x)
        return self.lin2(x1 * x_gate)


class TransformerBlock(nn.Module):
    """
    Configurable/modular MiMO-V2-Flash Transformer Block.
    TODO add complete arch like others

    Supports:
    - Global Attention (GA as they call, it's the usual causal attention, not full attention like a ViT) 
    or Sliding Window Attention (SWA)
    - Dense FFN or Mixture of Experts (MoE)
    """

    def __init__(self, cfg, layer_idx, use_sliding_window=True, use_moe=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_sliding_window = use_sliding_window
        self.use_moe = use_moe

        # SWA uses different number of KV groups (8 heads) vs GA (4 heads) per the MiMo-V2-Flash paper
        # our cfg should include "num_swa_kv_groups" and "num_ga_kv_groups" the way we're doing the logic
        if use_sliding_window:
            num_kv_groups = cfg["num_swa_kv_groups"]
        else:
            num_kv_groups = cfg["num_ga_kv_groups"]

        self.att = GroupedQueryAttentionWithSink(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=num_kv_groups,
            head_dim=cfg["head_dim"],
            use_sliding_window=use_sliding_window,
            window_size=cfg["window_size"],
            dtype=cfg["dtype"],
        )

        self.norm1 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.norm2 = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])

        # Feed Forward
        if use_moe:
            # reusing DeepSeekMoE but with 0 shared experts as per MiMo-V2-Flash paper
            self.feed_forward = DeepSeekMoE(
                cfg,
                num_experts=cfg["num_experts"],
                num_shared_experts=0,
                top_k=cfg["top_k"],
                scaling_factor=1.0,  # TODO adapt from DeepSeekMoE
            )
        else:
            self.feed_forward = FFN(cfg)

    def forward(self, x, mask, cos, sin, attn_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, attn_mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        return x
