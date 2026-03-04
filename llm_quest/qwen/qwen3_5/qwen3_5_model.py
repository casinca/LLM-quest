import torch
import torch.nn as nn

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.qwen.qwen3_5.qwen3_5_transformer_block import Qwen3_5TransformerBlock
from llm_quest.qwen.qwen3_next.qwen3_next_attention import ZeroCenteredRMSNorm


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


# Quick test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)
    model = Qwen3_5Model(config.QWEN3_5_08B_CONFIG)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    sample_input = torch.randint(0, 1000, (2, 10))  # b=2, seq_len=10
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0]}")
