import torch
import torch.nn as nn

from llm_quest.alignment.grpo.grpo_engine import PrefRewardCalculator
from llm_quest.gpt.gpt_transformer_block import LayerNorm, TransformerBlock


# This is a modified version of the GPT model, just like classification tasks, last_token_only=True is handy here for
# retrieving the last real token's score/reward and not a padding token.
class PreferenceRewardModel(nn.Module):
    """
    A reward model for preference data based on a GPT-2 model.
    It takes a sequence of token IDs as input and outputs a scalar reward.

    The reward can be calculated in different ways:
    - last_token_only: retrieve the last token's score/reward
    - hidden_state_pooling: mean pooling over the hidden states and then project to a scalar
    - logit_mean_pooling: project hidden states to a scalar and then mean pooling over the sequence length
    """

    def __init__(self, cfg):
        super().__init__()

        self.emb_dict = nn.Embedding(num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"])
        self.pos_emb_dict = nn.Embedding(cfg["context_length"], embedding_dim=cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # Using ModuleList instead of Sequential to properly pass attention mask
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for layer in range(cfg["n_layers"])])
        self.final_ln = LayerNorm(cfg["emb_dim"])

        # projecting output to get a scalar reward
        self.out = nn.Linear(cfg["emb_dim"], 1)

    def forward(self, x, attn_mask=None, reward_mask=None, last_token_only=False, hidden_states_pooling=False):
        b, seq_len = x.shape

        # shape (b, s) → (b, s, emb_dim)
        x = self.emb_dict(x)
        pos_emb = self.pos_emb_dict(torch.arange(seq_len, device=x.device))
        x = x + pos_emb
        x = self.dropout(x)

        for block in self.trf_blocks:
            x = block(x, attn_mask)

        x = self.final_ln(x)

        assert not (
            last_token_only and hidden_states_pooling
        ), "last_token_only and hidden_states_pooling cannot be True at the same time"

        # Retrieves the hidden state of the final valid token and not the last token (which could be a padding token)
        # Avoids unnecessary projection for all hidden states.
        if last_token_only:
            assert attn_mask is not None, "attn_mask are needed for last_token_only=True"
            # shape: (b, s, emb_dim) → slicing (b, emb_dim) → (b, 1)
            logits = PrefRewardCalculator.last_token_score(x, attn_mask, self.out)

        assert reward_mask is not None, "reward_mask is needed for hidden_states_pooling or scores_mean_pooling"
        if hidden_states_pooling:
            # shape: (b, s, emb_dim) → (b, emb_dim) → (b, 1)
            logits = PrefRewardCalculator.hidden_states_mean_pooling(x, reward_mask, self.out)

        else:
            # shape: (b, s, emb_dim) → (b, s, 1) → (b,)
            logits = self.out(x)
            logits = PrefRewardCalculator.scores_mean_pooling(logits, reward_mask)

        return logits
