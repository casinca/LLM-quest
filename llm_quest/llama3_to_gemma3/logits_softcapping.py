# This was originally used for Gemma2 in the:
# - Attention block
# - Model's output layer

import torch.nn.functional as F


def logits_softcap(softcap, logits):
    """
    A smooth & differentiable way to cap logits when they're excessively large (vs simply clipping).
    Attention logits being usually larger would require a higher "softcap" value compared to the output
    layer's logits.

    Args:
        softcap (float): The softcap value.
        logits (torch.Tensor): The logits to be capped.

    """
    return softcap * F.tanh(logits / softcap)
