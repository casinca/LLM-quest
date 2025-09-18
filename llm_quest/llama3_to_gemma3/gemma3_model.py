import torch.nn as nn

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.llama3_to_gemma3.gemma3_transformer_block import RMSNorm, TransformerBlock


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
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

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

        self.out_head.weight = self.emb_dict.weight  # weights tying

    # TODO attention mask (for now ghost argument for backward compatibility with evaluation _calc_loss_batch())
    def forward(self, x, attn_mask=None):
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin, self.swa_mask)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
