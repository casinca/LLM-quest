import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from llm_quest.common.hyper_connections.hyper_connections import (
    HyperConnectionPost,
    HyperConnectionPre,
    HyperConnectionRes,
)
from llm_quest.common.hyper_connections.manifold_hyper_connections import (
    MCHyperConnectionPost,
    MCHyperConnectionPre,
    MCHyperConnectionRes,
    MHCLiteRes,
)
from llm_quest.qwen.qwen3.qwen3_attention import PytorchRMSNorm
from llm_quest.qwen.qwen3.qwen3_model import Qwen3Model
from llm_quest.qwen.qwen3.qwen3_transformer_block import TransformerBlock


def _create_hyper_connection_set(emb_dim, expansion_rate, dtype):
    """Bundle for Classic Hyper-Connections: norm + res + pre + post for one sub-block (attn or ffn)."""
    return nn.ModuleDict(
        {
            "norm": PytorchRMSNorm(emb_dim, dtype=dtype),
            "res": HyperConnectionRes(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "pre": HyperConnectionPre(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "post": HyperConnectionPost(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
        }
    )


def _create_mhc_set(emb_dim, expansion_rate, dtype):
    """Bundle for DeepSeek's Manifold-Constrained Hyper-Connections: norm + res + pre + post for one sub-block (attn or
    ffn)."""
    return nn.ModuleDict(
        {
            "norm": PytorchRMSNorm(expansion_rate * emb_dim, dtype=dtype),
            "res": MCHyperConnectionRes(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "pre": MCHyperConnectionPre(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "post": MCHyperConnectionPost(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
        }
    )


def _create_mhc_lite_set(emb_dim, expansion_rate, dtype):
    """Bundle for mHC-lite: norm + res + pre + post for one sub-block (attn or ffn)."""
    return nn.ModuleDict(
        {
            "norm": PytorchRMSNorm(expansion_rate * emb_dim, dtype=dtype),
            "res": MHCLiteRes(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "pre": MCHyperConnectionPre(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
            "post": MCHyperConnectionPost(emb_dim, expansion_rate=expansion_rate, dtype=dtype),
        }
    )


class HyperQwen3TransformerBlock(TransformerBlock):
    """
        Qwen3Transformer Block with Hyper-Connections
        Flow for either Attention or FFN part of the trf block: in total done twice (attn+ffn) per trf block

                Input Stream x [B, S, n, emb]
                        ║
            ╔═══════════╬═════════════╗
            ║           ║             ║
            ║       if mhc/lite:      ║
            ║        [B, S, n*emb]    ║
            ║           ▼            ║
            ║      ┌───────────┐      ║
            ║      │  Routing  │      ║
            ║      │   Norm    │      ║
            ║      └─────┬─────┘      ║
            ▼           ║            ▼
        ┌─────────┐      ║       ┌─────────┐
        │ H_res @ x ◄═══╬════► │ H_pre @ x
        │ (Mix n) │      ║       │(Down n) │
        └───┬─────┘      ║       └────┬────┘
            ║            ║            │       [B, S, emb]
            ║            ║      ┌─────▼─────┐
            ║            ║      │ usual PreNorm
            ║            ║      │     +      │
            ║            ║      │ Attn or FFN│
            ║            ║      └─────┬──────┘
            ║            ║            │       [B, S, emb]
            ║            ║      ┌─────▼────┐
            ║            ╚════►│ H_post @ F(trf output)
            ║                   │  (expand back)
            ║                   └─────┬─────┘
            ║                         ║
            ╚════════════╦════════════╝       [B, S, n, emb]
                        ▼
                        +
                        ║
                    Output Stream

    Args:
        cfg (dict): Config dict containing hyperparams:
            - emb_dim (int): Embedding dimension
            - context_length (int): Context length for attention
            - n_heads (int): Number of attention heads
            - num_kv_groups (int): Number of key-value groups for GQA
            - head_dim (int): Head dimension for GQA
            - dtype (torch.dtype): Dtype of the weights, to change precision
        layer_idx (int): Layer index (used here for the KV cache)
        hc_type (str): Type of hyperconnections to use, "classic", "mhc" or "mhc-lite"
        expansion_rate (int): Expansion rate for the hyperconnections
    """

    def __init__(self, cfg, layer_idx, hc_type, expansion_rate=4):
        super().__init__(cfg, layer_idx)
        self.hc_type = hc_type

        if self.hc_type == "mhc":
            self.hc_attn = _create_mhc_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
            self.hc_ffn = _create_mhc_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
        elif self.hc_type == "mhc-lite":
            self.hc_attn = _create_mhc_lite_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
            self.hc_ffn = _create_mhc_lite_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
        elif self.hc_type == "classic":
            self.hc_attn = _create_hyper_connection_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
            self.hc_ffn = _create_hyper_connection_set(cfg["emb_dim"], expansion_rate, cfg["dtype"])
        else:
            raise ValueError(f"Invalid Hyper-Connections type: {self.hc_type}, must be 'mhc', 'mhc-lite' or 'classic'")

    def forward(self, x, mask, cos, sin, attn_mask=None, kv_cache=None, position_ids=None):
        b, s, exp_rate, emb_dim = x.shape

        # --- Attention part ---
        # the normalized input is used for the hyperconnections: generating H_res, H_pre, H_post
        # and is an addition/different from the usual pre-Norms from a classic trf block (that we also keep)

        # DeepSeek mHC is flattening the n streams, (exp_rate, emb_dim) → (exp_rate*emb_dim)
        # (it seems not contiguous, .view() throws an error, for now we use reshape at a small optim cost)
        x_norm_attn = (
            self.hc_attn["norm"](x.reshape(b, s, -1))
            if self.hc_type in ("mhc", "mhc-lite")
            else self.hc_attn["norm"](x)
        )
        # residual mixing
        residual = self.hc_attn["res"](x, x_norm_attn)

        # pre-mapping (down-project to a single stream for the trf block)
        x = self.hc_attn["pre"](x, x_norm_attn)  # shape (b, s, exps_rate, emb_dim) → (b, s, emb_dim)

        x = self.norm1(x)
        x = self.att(x, mask, cos, sin, attn_mask, kv_cache, position_ids)

        # post-mapping (broadcast back to expanded streams)
        x = self.hc_attn["post"](x, x_norm_attn)  # shape (b, s, emb_dim) → (b, s, exps_rate, emb_dim)
        x = x + residual

        # --- FFN part ---
        x_norm_ffn = (
            self.hc_ffn["norm"](x.reshape(b, s, -1)) if self.hc_type in ("mhc", "mhc-lite") else self.hc_ffn["norm"](x)
        )
        residual = self.hc_ffn["res"](x, x_norm_ffn)

        x = self.hc_ffn["pre"](x, x_norm_ffn)  # shape (b, s, exps_rate, emb_dim) → (b, s, emb_dim)

        x = self.norm2(x)
        x = self.ffn(x)

        x = self.hc_ffn["post"](x, x_norm_ffn)  # shape (b, s, emb_dim) → (b, s, exps_rate, emb_dim)
        x = x + residual
        return x


class HyperQwen3Model(Qwen3Model):
    """
    Dense Qwen3 model with HyperConnections.

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
        hc_type (str): Type of hyperconnections to use, "classic", "mhc" or "mhc-lite"
        expansion_rate (int): Expansion rate for the hyperconnections
    """

    def __init__(self, cfg, hc_type, expansion_rate=4):
        super().__init__(cfg)
        self.expansion_rate = expansion_rate

        # Override the transformer blocks with the HyperQwen3TransformerBlock
        self.trf_blocks = nn.ModuleList(
            [
                HyperQwen3TransformerBlock(cfg, layer_idx, hc_type=hc_type, expansion_rate=self.expansion_rate)
                for layer_idx in range(cfg["n_layers"])
            ]
        )

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
        # expanding the n streams (b, s, emb_dim) → (b, s, expansion_rate, emb_dim)
        x = x.unsqueeze(-2).expand(-1, -1, self.expansion_rate, -1)

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

        # collapsing the n streams (b, s, expansion_rate, emb_dim) → (b, s, emb_dim)
        # per the paper, we sum the stream (not averaging) because the model learns to split across streams, so features
        # should already be 1/n balanced across streams.
        x = x.sum(dim=-2)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


# quick inline test
if __name__ == "__main__":
    import torch

    import config

    torch.manual_seed(123)
    device = config.auto_device
    dtype = torch.bfloat16

    test_cfg = {
        "vocab_size": 50304,
        "context_length": 128,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 2,
        "num_kv_groups": 2,
        "hidden_dim": 512,
        "head_dim": 32,
        "rope_base": 10_000,
        "dtype": dtype,
        "tie_embeddings": True,
    }

    batch_size, seq_len = 2, 10
    hc_type = "mhc-lite"  # classic HC or DeepSeek's mHC or mHC-lite

    model = HyperQwen3Model(test_cfg, hc_type=hc_type, expansion_rate=4)
    model.to(device=device, dtype=dtype)

    input_ids = torch.randint(0, test_cfg["vocab_size"], (batch_size, seq_len), device=device)
    logits = model(input_ids)
    print(logits)
    print(logits.shape)
