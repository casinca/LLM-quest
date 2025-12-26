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
        # being explicit but all tied: emb_layer = main_emb_layer = out_layer = main_output_head
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


