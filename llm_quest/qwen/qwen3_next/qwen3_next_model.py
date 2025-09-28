import torch
import torch.nn as nn

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.qwen.qwen3_next.qwen3_next_attention import ZeroCenteredRMSNorm
from llm_quest.qwen.qwen3_next.qwen3_next_transformer_block import Qwen3NextTransformerBlock


class Qwen3NextModel(nn.Module):
    """
    Qwen3-Next model, similar to Qwen3 at this level of the architecture.
    - We pass the layer idx to the transformer block to determine which attention block to use
    - Zero-Centered RMSNorm is replacing RMSNorm

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
            [Qwen3NextTransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["n_layers"])],
        )

        self.final_norm = ZeroCenteredRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        mask, cos, sin = GlobalBuffers.get_buffers(
            ctx_len=cfg["context_length"],
            rope_base=cfg["rope_base"],
            head_dim=cfg["head_dim"],
            rotation_factor=cfg["partial_rope_factor"],
        )
        self.register_buffer("mask", ~mask)  # Inverted for compatibility with Pytorch's SDPA function
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, attn_mask=None):  # NOTE attention mask Placeholder
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


# Quick test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)
    model = Qwen3NextModel(config.SMALL_QWEN3_NEXT_CONFIG)

    sample_input = torch.randint(0, 1000, (2, 10))  # b=2, seq_len=10
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0]}")
