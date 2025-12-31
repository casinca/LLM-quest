import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_quest.common.buffers import GlobalBuffers
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.xiaomi.mimo_v2_flash_transformer_block import TransformerBlock


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) Module, similar to the DeepSeek V3 MTP module implemented in `deepseek_model.py`
    Lightweight block: SWA + Dense FFN per MiMo-V2-Flash paper.

    Shared embeddings and output head with Main Model.
    """

    def __init__(self, cfg, main_output_head):
        super().__init__()
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

    def forward(self, x_embeds, h_prev, mask, cos, sin):
        """
        x_embeds: input embeddings for this MTP step (shifted)
        h_prev: hidden states from previous step (Main Model or previous MTP)
        """
        x = self.rms_input(x_embeds)
        h_prev = self.rms_h_prev(h_prev)

        combined = torch.cat([x, h_prev], dim=-1)
        x = self.down_proj(combined)

        h_curr = self.trf_block(x, mask, cos, sin)
        # apply final norm before shared head (since head expects normalized input for consistency with main model)
        logits = self.out_layer(self.final_norm(h_curr))

        return logits, h_curr


class MainModel(nn.Module):
    """
    MiMo Main Model Backbone (without MTP).
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


# NOTE: The MTP logic is slightly different from DSV3 MTP, but is more efficient:
# Instead of having, on top of the main model input/target, k prepared pre-shifted inputs/targets pairs for each MTP,
# here we just have one input/target pair that we sequentially shrink at each step for MTPs
class MiMoModel(nn.Module):
    """
    Full MiMo-V2-Flash Model with MTP.
    """

    def __init__(self, cfg):
        super().__init__()
        self.main_model = MainModel(cfg)
        self.mtp_depth = cfg.get("mtp_depth", 0)
        self.mtp_coeff = cfg.get("mtp_loss_coeff", 0.0)

        self.mtp_modules = nn.ModuleList([MTPModule(cfg, self.main_model.out_head) for _ in range(self.mtp_depth)])

    def forward(self, x, targets=None, training=True):
        """
        x: (b, s) input tokens
        targets: (b, s) target tokens, should be already shifted by 1 (aligned with logits)
        """

        logits, h_prev = self.main_model(x)  # (b, s, vocab_size) and (b, s, emb_dim)

        if not training or targets is None:
            return logits

        main_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        if self.mtp_depth == 0:
            return main_loss

        # --- MTP logic ---
        # Buffers for MTP (always SWA)
        mtp_mask = self.main_model.mask_swa
        mtp_cos = self.main_model.cos_swa
        mtp_sin = self.main_model.sin_swa

        # retrieving input embeddings from the main model (tied with MTP modules)
        x_embeds = self.main_model.emb_layer(x)

        mtp_loss_total = 0
        for i, mtp_module in enumerate(self.mtp_modules):
            # Input embeddings prep:
            k = i + 1  # MTP slicing index
            # since mtp predicts t+k, we slice to start at k and remove the last token (no label for last token)
            mtp_slice = x_embeds[:, k:-1]  # (b, s-k-1, emb_dim)
            mtp_target = x[:, k + 1 :]  # aligning targets with mtp_slice

            # Hidden states prep:
            if k == 1:
                # for the first MTP, h_prev is the main model h_states, so its shape is s, we remove the last 2 h_states
                # to match mtp_slice shape (s-k-1)
                h_slice = h_prev[:, :-2]
                # for the rest, h_prev is from previous MTP h_states, so its shape is s-k, we just remove the last
                # hidden state (no label, same as mtp_slice :-1)
            else:
                h_slice = h_prev[:, :-1]

            seq_len = h_slice.shape[1]
            if seq_len == 0:  # shouldn't happen quick break
                print(f"sequence length is 0, breaking at k={k}")
                break
            # in DSV3 MTP we werent shrinking but had already same length preshifted but need investigation # TODO
            # slice cos/sin to match shrinking sequence length at each step for MTPs attn block
            curr_cos = mtp_cos[:seq_len]
            curr_sin = mtp_sin[:seq_len]

            mtp_logits, h_curr = mtp_module(mtp_slice, h_slice, mtp_mask, curr_cos, curr_sin)

            mtp_loss = F.cross_entropy(mtp_logits.flatten(0, 1), mtp_target.flatten())
            mtp_loss_total += mtp_loss

            h_prev = h_curr  # passing hidden states to the next MTP module

        total_loss = main_loss + (self.mtp_coeff / self.mtp_depth) * mtp_loss_total  # DeepSeekV3 style average mtp loss

        return total_loss
