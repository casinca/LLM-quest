import torch
import torch.nn as nn

from llm_quest.gpt.gpt_transformer_block import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    """
    A gpt2 model implementation.

    This model originally followed the architecture described in the GPT papers, but later was modified for KVCache,
    Multi-modal early fusion, attention mask and retrieving last token logits, while preserving the original
    architecture.

    Originally consisting of:
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
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx=layer_idx) for layer_idx in range(cfg["n_layers"])]
        )
        self.final_ln = LayerNorm(cfg["emb_dim"])
        # projecting output to vocab_size to get logits
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(
        self,
        x,
        attn_mask=None,
        kv_cache=None,
        last_token_only=False,
        input_embedded=False,
        position_ids=None,
    ):
        """
        Forward pass of the GPT model.

        Args:
            x (torch.Tensor): Input tensor. Shape (b, seq_len) containing token IDs if
                input_embedded is False, or (b, seq_len, emb_dim) if input_embedded is True.
            attn_mask (torch.Tensor, optional): Attention mask of shape (b, seq_len).
                Contains True for valid tokens and False for padding tokens.
            kv_cache (KVCache, optional): Key-Value cache object for efficient generation.
            last_token_only (bool): If True, returns logits only for the last valid token.
                Defaults to False.
            input_embedded (bool): Whether the input x is already embedded (for multimodal early fusion).
                                Defaults to False.
            position_ids (torch.Tensor, optional): Precomputed position IDs of shape (b, 1)
                or (b, seq_len). If None, they are computed from the kv_cache start position.

        Returns:
            torch.Tensor: Logits of shape (b, seq_len, vocab_size) if last_token_only is False,
                or (b, vocab_size) if last_token_only is True.
        """
        # Input for Multimodal Early fusion training is already embedded: (b, num_patches + 1 + seq_len, emb_dim)
        # thus bypassing token and positional embedding
        if not input_embedded:
            b, seq_len = x.shape
            # shape (b, s) → (b, s, emb_dim)
            x = self.emb_dict(x)

            # --- Modif for positional embeddings ---
            # we use precomputed position_ids for positional embeddings if provided (avoids fake padding positions)
            # otherwise for single generation, we can compute them from the KVCache start position
            if position_ids is None:
                past_len = 0
                if kv_cache is not None:
                    # need to add the length to correctly offset the positions for the current tokens
                    past_len = kv_cache.start_pos
                position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)  # (1, s)

            pos_emb = self.pos_emb_dict(position_ids)  # same device as input

            # old way:
            # pos tensor has the same length as the current batch seq len
            # and not fixed ctx_len size, thus having a dynamic shape per batch
            # pos_emb = self.pos_emb_dict(torch.arange(seq_len, device=x.device))  # same device as input

            x = x + pos_emb

        x = self.dropout(x)

        # Pass through each transformer block, providing both x, attn_mask, and kv_cache
        for block in self.trf_blocks:
            x = block(x, attn_mask, kv_cache)

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
