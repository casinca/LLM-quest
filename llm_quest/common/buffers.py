import torch

from llm_quest.common.rope import RoPE


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
    def get_buffers(ctx_len, rope_base, head_dim, smooth_scaling_cfg=None, rotation_factor=1.0):
        key = (ctx_len, rope_base, head_dim)

        if key not in GlobalBuffers._buffer:
            mask = torch.triu(torch.ones(ctx_len, ctx_len, dtype=torch.bool), diagonal=1)
            cos, sin = RoPE.compute_angles(
                base=rope_base,
                head_dim=head_dim,
                ctx_len=ctx_len,
                smooth_scaling_cfg=smooth_scaling_cfg,
                rotation_factor=rotation_factor,
            )

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
