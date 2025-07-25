import torch
import torch.nn as nn

from llm_quest.gpt.gpt_transformer_block import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    """
    A gpt model implementation.

    This model follows the architecture described in the GPT papers, consisting of:
    - Token embeddings
    - Positional embeddings
    - Multiple transformer blocks
    - Layer normalization
    - Output projection to vocabulary size

    The model takes sequences of token IDs as input and outputs logits over the vocabulary
    for next-token prediction.

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.emb_dict = nn.Embedding(num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"])
        self.pos_emb_dict = nn.Embedding(cfg["context_length"], embedding_dim=cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # Using ModuleList instead of Sequential to properly pass attention mask
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for layer in range(cfg["n_layers"])])
        self.final_ln = LayerNorm(cfg["emb_dim"])
        # projecting output to vocab_size to get logits
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x, attn_mask=None, last_token_only=False):
        b, seq_len = x.shape

        # shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)
        # pos tensor has the same length as the current batch seq len
        # and not fixed ctx_len size, thus having a dynamic shape per batch
        pos_emb = self.pos_emb_dict(torch.arange(seq_len, device=x.device))  # same device as input
        x = x + pos_emb
        x = self.dropout(x)

        # Pass through each transformer block, providing both x and attn_mask
        for block in self.trf_blocks:
            x = block(x, attn_mask)

        x = self.final_ln(x)

        # Retrieves the hidden state of the final valid token and not the last token (which could be a padding token)
        # Avoids unnecessary projection for all hidden states.
        if last_token_only:
            assert attn_mask is not None, "attn_mask are needed for last_token_only=True"
            seq_lengths = attn_mask.sum(dim=-1)
            # shape: (b, s, emb_dim) → slicing (b, emb_dim) → (b, vocab_size)
            logits = self.out(x[torch.arange(b), seq_lengths - 1, :])
        else:
            # apply output layer to all hidden states
            # shape: (b, s, emb_dim) → (b, s, vocab_size)
            logits = self.out(x)

        return logits
