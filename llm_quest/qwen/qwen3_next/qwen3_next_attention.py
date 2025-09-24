import torch
import torch.nn as nn


class ZeroCenteredRMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm used in Qwen3-Next
    This is a RMSNorm with 0 centered weights (instead of classic 1s initialization)

    To make the forward pass work, we add 1 to the weights to get the correct scaling back.
    This is a trick to better adapt L2 regularization with RMSNorm.

    Note: We also do the full forward in fp32

    Args:
        emb_dim (int): The dimension of the embeddings to "normalize" over.
        eps (float): The epsilon value to avoid division by zero.
        dtype (torch.dtype, optional): Data type for the weights. Defaults to None.
    """

    def __init__(self, emb_dim, eps=1e-6, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(emb_dim, dtype=dtype))  # 0 centered weights
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x * rms * (1 + self.scale)).to(input_dtype)  # fullcast to fp32 before returning to input dtype
