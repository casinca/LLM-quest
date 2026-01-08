import torch
import torch.nn as nn

# separating each hyper-connection to 3 separate classes (Res, Pre, Post) instead of one class because
# they are all initialized differently and there is 3 different forward passes for each of them


class HyperConnectionRes(nn.Module):
    """
    Classic Hyper-connection for residual stream as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 3 and 5): https://arxiv.org/abs/2512.24880
    - Hyper-Connections (HC) paper: https://arxiv.org/abs/2409.19606

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The activation function class, default is Tanh per the mHC paper
        norm_cls (nn.Module): The normalization function class, default is RMSNorm per the mHC paper

    Returns:
        x: The mixed residual streams, shape: (b, seq_len, exps_rate, emb_dim)
    """

    def __init__(self, emb_dim, expansion_rate=4, add_static_mapping=True, activation_cls=nn.Tanh, norm_cls=nn.RMSNorm):
        super().__init__()

        self.norm = norm_cls(emb_dim)
        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01]))

        # dynamic mapping (theta_res): project to n streams
        self.linear = nn.Linear(emb_dim, expansion_rate, bias=False)
        # The HC paper init dynamic mapping weights for theta_res as 0 (HC paper p.4 section 2.3) but Pytorch nn.linear
        # is doing Kaiming init by default so we need to zero init
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_res) are initialized as the identity matrix (HC paper), since dynamic mapping weights
        # are initialized to 0, overall, this makes the hyper-connection start with an untouched stream (like a classic
        # residual connection): (0 + I) @ xl = xl
        self.bias = nn.Parameter(torch.eye(expansion_rate)) if add_static_mapping else None  # (exps_rate, exps_rate)

    # TODO mention diff with DeepSeek: expansion stream are flattened, (n, emb_dim) -> (n*emb_dim) no need for transpose
    def residual_matrix(self, x):
        """
        Generates the residual mapping/matrix H_res
        This is our small "HyperNet" where we generate on the fly the weights H_res to mix with the residual stream

        Args:
            x: The n untouched/identity/residual streams input, shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The residual mapping/matrix H_res, shape: (b, seq_len, exps_rate, exps_rate)
        """
        x = self.norm(x)
        x = self.linear(x)  # apply dynamic mapping
        # Pytorch nn.linear is doing Wx (as XW^T) but we want WX^T (eq 5), therefore we need to transpose the linear
        # output as (XW^T)^T = WX^T
        x = x.mT

        # activate and scale
        x = self.activation(x) * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        return x

    def forward(self, x):
        """
        Apply/mix the residual mapping/matrix H_res to the residual stream

        Args:
            x: The n untouched/identity/residual streams input, shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The n mixed streams after applying the residual mapping H_res, shape: (b, seq_len, exps_rate, emb_dim)
        """
        x = self.residual_matrix(x) @ x

        return x

