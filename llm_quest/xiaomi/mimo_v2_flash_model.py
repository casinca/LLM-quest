import torch
import torch.nn as nn

from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.xiaomi.mimo_v2_flash_transformer_block import TransformerBlock


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) Module, similar to the DeepSeek V3 MTP module implemented in `deepseek_model.py`
    Lightweight block: SWA + Dense FFN per MiMo-V2-Flash paper.

    Shared embeddings and output head with Main Model.
    """

    def __init__(self, cfg, main_emb_layer, main_output_head):
        super().__init__()
        self.emb_layer = main_emb_layer
        self.out_layer = main_output_head

        # MTP norms
        self.rms_h_prev = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.rms_input = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        # Final Norm before shared LM head (missing from the paper)
        self.final_norm = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])

        # projection for combining current input and previous hidden state
        self.down_proj = nn.Linear(2 * cfg["emb_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

        # MTP uses SWA and Dense FFN
        self.trf_block = TransformerBlock(cfg, layer_idx=0, use_sliding_window=True, use_moe=False)

    def forward(self, x, h_prev, mask, cos, sin):
        """
        x: input tokens for this MTP step (shifted) # TODO need to see if same as we did for DSV3 or not
        h_prev: hidden states from previous step (Main Model or previous MTP)
        """
        x = self.emb_layer(x)
        x = self.rms_input(x)
        h_prev = self.rms_h_prev(h_prev)

        combined = torch.cat([x, h_prev], dim=-1)
        x = self.down_proj(combined)

        h_curr = self.trf_block(x, mask, cos, sin)
        # apply final norm before shared head (since head expects normalized input for consistency with main model)
        logits = self.out_layer(self.final_norm(h_curr))

        return logits, h_curr


class MainModel(nn.Module):
    """
    MiMO Main Model Backbone (without MTP).
    Layer 0: GA + Dense FFN
    Layers 1+: Hybrid Blocks (5 SWA + 1 GA) with MoE
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.emb_layer = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.layers = nn.ModuleList()
        for i in range(cfg["n_layers"]):
            if i == 0:
                # First layer: GA + Dense FFN
                use_sw = False
                use_moe = False
            else:
                use_moe = True  # All hybrid layers use MoE
                use_sw = True if (i + 1) % cfg["hybrid_ratio"] else False  # 5 SWA : 1 GA

            self.layers.append(TransformerBlock(cfg, layer_idx=i, use_sliding_window=use_sw, use_moe=use_moe))

        self.final_norm = PytorchRMSNorm(cfg["emb_dim"], dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Buffers for SWA and GA
        # we need two sets of RoPE buffers: one for SWA (default base) and one for GA (high base)
        self.rope_base_swa = cfg.get("rope_base", 10000)
        self.rope_base_ga = cfg.get("rope_base_ga", 640000)  # default from paper for 32k training

        mask_swa = GlobalBuffers.get_swa_mask(cfg["context_length"], cfg["window_size"])
        cos_swa, sin_swa = GlobalBuffers.get_rope_params(
            ctx_len=cfg["context_length"],
            rope_base=self.rope_base_swa,
            head_dim=cfg["head_dim"],
            rotation_factor=cfg["partial_rope_factor"],
        )
        self.register_buffer("mask_swa", mask_swa)
        self.register_buffer("cos_swa", cos_swa)
        self.register_buffer("sin_swa", sin_swa)

        mask_ga = GlobalBuffers.get_causal_mask(cfg["context_length"])
        cos_ga, sin_ga = GlobalBuffers.get_rope_params(
            ctx_len=cfg["context_length"],
            rope_base=self.rope_base_ga,
            head_dim=cfg["head_dim"],
            rotation_factor=cfg["partial_rope_factor"],
        )
        self.register_buffer("mask_ga", mask_ga)
        self.register_buffer("cos_ga", cos_ga)
        self.register_buffer("sin_ga", sin_ga)

    def forward(self, x, attn_mask=None):
        x = self.emb_layer(x)

        for i, layer in enumerate(self.layers):
            if layer.use_sliding_window:
                mask, cos, sin = self.mask_swa, self.cos_swa, self.sin_swa
            else:
                mask, cos, sin = self.mask_ga, self.cos_ga, self.sin_ga

            x = layer(x, mask, cos, sin, attn_mask)

        h_final = x
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits, h_final


