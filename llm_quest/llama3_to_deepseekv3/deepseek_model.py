import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from llm_quest.common.buffers import GlobalBuffers
from llm_quest.engine import global_loss
from llm_quest.gpt_to_llama3.llama_transformer_block import RMSNorm
from llm_quest.llama3_to_deepseekv3.deepseek_transformer_block import TransformerBlock


class MTPModule(nn.Module):
    """
    This class represents the Multi Token Prediction (MTP) module (a single LLM layer).
    The k-th MTP modules are stacked on top of each other after the main model.

    Args:
        cfg (dict): Configuration dictionary containing hyperparameters.
        main_emb_layer (nn.Module): The embedding layer of the main model.
        main_output_head (nn.Module): The output head of the main model.
    """

    def __init__(self, cfg, main_emb_layer, main_output_head):
        super().__init__()

        self.emb_layer = main_emb_layer  # shared
        self.rms_h_prev = RMSNorm(cfg["emb_dim"])
        self.rms_input = RMSNorm(cfg["emb_dim"])
        self.down_proj = nn.Linear(2 * cfg["emb_dim"], cfg["emb_dim"])
        self.trf_block = TransformerBlock(cfg, layer=0)  # MTP trf block should have a FFN, not MoE. Layers < 3 = FFN
        self.out_layer = main_output_head  # shared

    def forward(self, x, h_prev, mask, cos, sin):
        """
        args:
            x (torch.Tensor): Input tensor.
            h_prev (torch.Tensor): Hidden states from: the main model or previous MTP module.
        """

        x = self.emb_layer(x)
        x = self.rms_input(x)
        h_prev = self.rms_h_prev(h_prev)

        x = self.down_proj(torch.cat([x, h_prev], dim=-1))

        h_curr = self.trf_block(x, mask, cos, sin)
        logits = self.out_layer(x)

        return logits, h_curr


class MainModel(nn.Module):
    """
    This class represents the Main LLM model of DeepSeek V3 without MTP.
    """

    def __init__(self, cfg):
        super().__init__()

        self.emb_layer = nn.Embedding(
            num_embeddings=cfg["vocab_size"],
            embedding_dim=cfg["emb_dim"],
            dtype=cfg["dtype"],
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer) for layer in range(cfg["n_layers"])],
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # x shape (b, s) → (b, s, emb_dim)
        x = self.emb_layer(x)

        for block in self.trf_blocks:
            x = block(x, mask, cos, sin)

        h_curr = x
        x = self.final_norm(h_curr)
        logits = self.out_layer(x)

        return logits, h_curr  # we return logits + hidden states


class DeepSeekV3Model(nn.Module):
    """
    This class is the complete DeepSeek V3 model combining the Main LLM model and 'depth' MTP modules.
    """

    def __init__(self, cfg):
        super().__init__()
        self.coeff = cfg["mtp_loss_coeff"]
        self.depth = cfg["mtp_depth"]
        self.main_model = MainModel(cfg)
        # we share embeddings and output head for the Main model + MTP modules
        self.mtp_modules = nn.ModuleList(
            [MTPModule(cfg, self.main_model.emb_layer, self.main_model.out_layer) for _ in range(self.depth)]
        )

        # Initialize buffers
        # Not using extended context length scaling(smooth_scaling_cfg) for pretraining
        mask = GlobalBuffers.get_causal_mask(cfg["context_length"])
        cos, sin = GlobalBuffers.get_rope_params(
            cfg["context_length"],
            cfg["rope_base"],
            cfg["emb_dim"] // cfg["n_heads"] // 2,  # decoupled Q and K = head_dim / 2
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, y, shifted_x, shifted_y):

        logits, h_prev = self.main_model(x, self.mask, self.cos, self.sin)

        # they didn't use MTPs as a form of speculative decoding for inference (unlike Xiaomi MiMo-V2-Flash)
        if not self.training and y is None:
            return logits

        main_loss = global_loss(logits, y, model=self.main_model)

        # eval mode check: return main model loss only (no MTP losses)
        if not self.training or self.depth == 0:
            return main_loss

        mtp_losses = 0
        for k, mtp_module in enumerate(self.mtp_modules):

            s_x, s_y = shifted_x[k].to("cuda"), shifted_y[k].to("cuda")

            mtp_logits, h_curr = mtp_module(s_x, h_prev, self.mask, self.cos, self.sin)
            mtp_loss = F.cross_entropy(mtp_logits.flatten(0, 1), s_y.flatten())
            mtp_losses += mtp_loss
            h_prev = h_curr  # passing hidden states to the next MTP module

        total_loss = main_loss + (self.coeff / self.depth) * mtp_losses

        return total_loss


# quick test
if __name__ == "__main__":
    # visual
    inputs = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 50256, 50256],
        [9, 10, 11, 12, 50256],
    ]
    targets = [
        [2, 3, 4, 5, 50256],
        [7, 8, 50256, -100, -100],
        [10, 11, 12, 50256, -100],
    ]

    shifted_inputs1 = [
        [2, 3, 4, 5, 50256],
        [7, 8, 50256, 50256, 50256],
        [10, 11, 12, 50256, 50256],
    ]

    shifted_targets1 = [
        [3, 4, 5, 50256, -100],
        [8, 50256, -100, -100, -100],
        [11, 12, 50256, -100, -100],
    ]

    shifted_inputs2 = [
        [3, 4, 5, 50256, 50256],
        [8, 50256, 50256, 50256, 50256],
        [11, 12, 50256, 50256, 50256],
    ]

    shifted_targets2 = [
        [4, 5, 50256, -100, -100],
        [50256, -100, -100, -100, -100],
        [12, 50256, -100, -100, -100],
    ]

    device = config.auto_device

    tensor_input = torch.tensor(inputs).to(device)
    tensor_target = torch.tensor(targets).to(device)

    shifted_inputs, shifted_targets = [], []
    shifted_inputs.append(torch.tensor(shifted_inputs1).to(device))
    shifted_inputs.append(torch.tensor(shifted_inputs2).to(device))
    shifted_targets.append(torch.tensor(shifted_targets1).to(device))
    shifted_targets.append(torch.tensor(shifted_targets2).to(device))

    print(shifted_inputs)
    print(shifted_targets)

    dsv3 = DeepSeekV3Model(config.DEEPSEEK_SMALL_CONFIG)
    dsv3.to(device)
    loss = dsv3(tensor_input, tensor_target, shifted_inputs, shifted_targets)
    print(loss)

    # main_model = MainModel(config.DEEPSEEK_SMALL_CONFIG)
    # main_model.to(device)

    # logits, h = main_model(tensor_input)
    # loss = global_loss(logits, tensor_target, model=main_model)
