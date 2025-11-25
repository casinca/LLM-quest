import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.qwen.qwen3.qwen3_transformer_block import MoETransformerBlock, TransformerBlock


class Qwen3Model(nn.Module):
    """
    A Qwen3 model implementation following the different architectures:

    Supports both dense and MoE variants based on the config cfg["model_type"]

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()
        self.tie_embeddings = cfg["tie_embeddings"]  # attribute to be checked by the weight loading function
        # Qwen cfgs are for inference, so this key is not originally present, it will be False by default
        self.gradient_checkpointing = cfg.get("gradient_checkpointing", False)
        self.emb_dict = nn.Embedding(
            num_embeddings=cfg["vocab_size"],
            embedding_dim=cfg["emb_dim"],
            dtype=cfg["dtype"],
        )

        # Choose transformer block type based on model configuration
        if cfg["model_type"] == "moe":
            self.trf_blocks = nn.ModuleList(
                [MoETransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["n_layers"])],
            )
        else:  # dense
            self.trf_blocks = nn.ModuleList(
                [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["n_layers"])],
            )

        self.final_norm = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])

        # Weight tying based on model configuration
        # this part is only useful for either: pretraining or reducing memory allocation before loading weights
        if self.tie_embeddings:
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"], device="meta")
            assert (
                self.tie_embeddings and self.emb_dict.weight.shape == self.out_head.weight.shape
            ), "Shape mismatch for weight tying"
            self.out_head.weight = self.emb_dict.weight  # weights tying
        else:
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Initialize buffers for RoPE and causal mask
        mask, cos, sin = GlobalBuffers.get_buffers(
            cfg["context_length"],
            cfg["rope_base"],
            cfg["head_dim"],
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, attn_mask=None, kv_cache=None, position_ids=None):
        """
        args:
            x: (b, s)
            attn_mask: (b, s) Attention mask (passed as True = real tokens)
            kv_cache: KVCache instance/object
            position_ids: (b, s/1) (long tensor), containing the position of each token in the sequence
        """
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)

        # Use gradient checkpointing during training, if enabled. Not compatible with inference (no backprop needed)
        # checkpoint() recomputes the forward during the backward pass to save memory, at the cost of speed.
        use_checkpointing = self.gradient_checkpointing and self.training and kv_cache is None

        for block in self.trf_blocks:
            if use_checkpointing:
                x = checkpoint(
                    block,
                    x,
                    self.mask,
                    self.cos,
                    self.sin,
                    attn_mask,
                    kv_cache,
                    position_ids,
                    use_reentrant=False,  # needs to be explicit since Pytorch 2.9 and False is recommended
                )
            else:
                x = block(x, self.mask, self.cos, self.sin, attn_mask, kv_cache, position_ids)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


# Quick test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)
    model = Qwen3Model(config.qwen3_config_creator("temp_moe"))

    sample_input = torch.randint(0, 1000, (2, 10))  # b=2, seq_len=10
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output[0]}")
