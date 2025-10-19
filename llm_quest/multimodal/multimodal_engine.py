import torch
import torch.nn.functional as F


def get_embeddings(text_input, model):
    """
    helper function to return token + positional embeddings for the text input

    Args:
        text_input(torch.tensor): text input ids, shape (batch_size, seq_len)
        model: gpt2 model or no RoPE model

    returns:
        (torch.tensor): Combined token + positional embeddings, shape (batch_size, seq_len, emb_dim)
    """
    token_embeddings = model.emb_dict(text_input)
    positions = torch.arange(token_embeddings.shape[1], dtype=torch.long, device=token_embeddings.device)
    position_embeddings = model.pos_emb_dict(positions)

    return token_embeddings + position_embeddings


def multimodal_loss(logits, labels, text_attention_mask, num_vision_tokens):
    """
    Classic NTP with a CE loss
    We want to predict the text tokens given the image + text context
    so image tokens/patches+padding are ignored, text_tokens are predicted

    Args:
        logits: (torch.tensor): shape (batch, seq_len, vocab_size)
        labels: (torch.tensor): shape (batch, seq_len)
        text_attention_mask: (torch.tensor): shape (batch, seq_len)
        num_vision_tokens: (int): number of vision tokens in the input
    """

    shifted_logits = logits[:, num_vision_tokens - 1 : -1, :]  # the last vision token predicts the first text token
    labels = labels.masked_fill(text_attention_mask == 0, -100)  # already aligned with shifted_logits

    loss = F.cross_entropy(shifted_logits.flatten(0, 1), labels.flatten(), ignore_index=-100)

    return loss
