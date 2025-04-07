import torch
import torch.nn as nn

from llm_from_scratch.GPT_to_Llama32.llama_transformer_block import RMSNorm, TransformerBlock
from llm_from_scratch.rope import RoPE


# replace: LayerNorm with RMSNorm
# remove:  - absolute positional embeddings
#          - dropout
# add: - weights tying
#      - dtype setting
#      - implementing a global cache class for RoPE param + attention mask
#      (similar to @rasbt optimized imp but I'm keeping a buffer class for separation of concerns)
#
# Note: the way weights are tied, assume we won't load both pretrained tok embs and out embs weights
class GlobalBuffers:
    """
    GlobalBuffers is a class that implements a global cache for RoPE parameters and attention masks to avoid redundant
    computations across different transformer blocks.

    Attributes:
        _buffer (dict):
            A class-level dictionary that stores the precomputed mask, cos, and sin values indexed by a tuple of
            (ctx_len, rope_base, head_dim).
    """

    _buffer = {}

    @staticmethod
    def get_buffers(ctx_len, rope_base, head_dim, smooth_scaling_cfg=None):
        key = (ctx_len, rope_base, head_dim)

        if key not in GlobalBuffers._buffer:
            mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
            cos, sin = RoPE.compute_angles(rope_base, head_dim, ctx_len, smooth_scaling_cfg)

            GlobalBuffers._buffer[key] = (mask, cos, sin)

        return GlobalBuffers._buffer[key]


class Llama3Model(nn.Module):
    """
    A GPT to Llama 3.2 model conversion implementation.

    This model follows the architecture described in the Llama paper, consisting of:
    - Token embeddings
    - Multiple transformer blocks
    - RMS normalization
    - Output projection to vocabulary size (weights tied to emb weights)

    The model takes sequences of token IDs as input and outputs logits over the vocabulary
    for next-token prediction.

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.emb_dict = nn.Embedding(
            num_embeddings=cfg["vocab_size"],
            embedding_dim=cfg["emb_dim"],
            dtype=cfg["dtype"],
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for layer in range(cfg["n_layers"])],
        )
        self.final_ln = RMSNorm(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Initialize buffers
        # Not using extended context length scaling(smooth_scaling_cfg) for pretraining
        mask, cos, sin = GlobalBuffers.get_buffers(
            cfg["context_length"],
            cfg["rope_base"],
            cfg["emb_dim"] // cfg["n_heads"],
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        assert self.emb_dict.weight.shape == self.out.weight.shape, "Shape mismatch for weight tying"
        self.emb_dict.weight = self.out.weight  # weights tying

    def forward(self, x):
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)
        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)
        x = self.final_ln(x)
        logits = self.out(x)
        return logits
