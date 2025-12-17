import torch
import torch.nn as nn


class SquaredReLU(nn.Module):
    """
    A ReLU activation but squared.
    paper: https://arxiv.org/abs/2109.08668

    Nvidia Nemotron 3 uses this activation function in the FFN.
    """

    def forward(self, x):
        return torch.square(torch.relu(x))
