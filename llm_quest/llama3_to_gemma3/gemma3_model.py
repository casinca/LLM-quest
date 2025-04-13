import torch
import torch.nn as nn

from llm_quest.llama3_to_gemma3.gemma3_transformer_block import RMSNorm, TransformerBlock
from llm_quest.rope import RoPE


class GlobalBuffers:
    """
    GlobalBuffers is a class that implements a global cache for RoPE parameters and masks to avoid redundant
    computations across different transformer blocks.

    Attributes:
        _buffer (dict):
            A class-level dictionary that stores the precomputed attention mask, cos, and sin values
        _swa_buffer (dict):
            A class-level dictionary that stores the precomputed sliding window attention mask.
    """

    _buffer = {}
    _swa_buffer = {}

    @staticmethod
    def get_buffers(ctx_len, rope_base, head_dim, smooth_scaling_cfg=None):
        key = (ctx_len, rope_base, head_dim)

        if key not in GlobalBuffers._buffer:
            mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1)
            cos, sin = RoPE.compute_angles(rope_base, head_dim, ctx_len, smooth_scaling_cfg)

            GlobalBuffers._buffer[key] = (mask, cos, sin)

        return GlobalBuffers._buffer[key]

    @staticmethod
    def get_swa_buffers(ctx_len, window_size):

        key = (ctx_len, window_size)

        if key not in GlobalBuffers._swa_buffer:
            k_range = torch.arange(window_size)
            i_range = torch.arange(ctx_len).unsqueeze(-1)
            swa_mask = k_range < (window_size - 1 - i_range)

            GlobalBuffers._swa_buffer[key] = swa_mask

        return GlobalBuffers._swa_buffer[key]


class Gemma3Model(nn.Module):
    """
    A Llama3.2 to Gemma3 model conversion implementation.

    Identical to Llama3.2 except we remove for Gemma3:
        - weight tying (optional, unclear)

    from Gemma2 we remove:
        - logits softcapping


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
            [TransformerBlock(cfg, layer) for layer in range(cfg["n_layers"])],
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

        swa_mask = GlobalBuffers.get_swa_buffers(cfg["context_length"], cfg["window_size"])
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("swa_mask", swa_mask)

        self.emb_dict.weight = self.out.weight  # weights tying

    def forward(self, x):
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)
        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin, self.swa_mask)
        x = self.final_ln(x)
        logits = self.out(x)

        return logits
