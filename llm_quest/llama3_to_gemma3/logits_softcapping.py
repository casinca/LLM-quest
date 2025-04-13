# This was originally used for Gemma2 in the:
# - Attention block
# - Model's output layer
import torch
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


if __name__ == "__main__":
    # x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    #
    # windows = x.unfold(dimension=0, size=2, step=2)
    # print(windows)

    torch.manual_seed(123)
    b = 1
    num_heads = 2
    seq_len = 5
    window_size = 3
    head_dim = 4
    keys = torch.randn(b, num_heads, seq_len, head_dim)

    print(keys)

    pad_size = window_size - 1
    padded_keys = F.pad(keys, (0, 0, pad_size, 0))  # (left, right, top, bottom)

    # Calculate correct strides - critical fix!
    original_strides = padded_keys.stride()  # (batch, heads, seq, dim)
    new_stride = (
        original_strides[0],  # batch stride
        original_strides[1],  # heads stride
        original_strides[2],  # sequence stride (1 position per window step)
        original_strides[2],  # window stride (1 position per element)
        original_strides[3],  # dim stride
    )

    # Create windows - now correctly sliding horizontally
    key_windows = padded_keys.as_strided(size=(b, num_heads, seq_len, window_size, head_dim), stride=new_stride)

    print(key_windows)

    print(torch.arange(5))
    print(torch.arange(5).unsqueeze(0))
