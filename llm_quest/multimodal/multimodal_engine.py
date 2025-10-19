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
