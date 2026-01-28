# This Manifold-Constrained Hyper-Connections (mHC) is based on the hyper_connections.py file, where we add the mHC
# modifications:
# Even if the global logic is the same, DeepSeek flatten the n expanded streams from a matrix (n x C) to a
# row vector (1 x nC) therefore we need to change the projections for all the hyper-connections.
# They mention doing that to "preserve full context information".
#
# separating each hyper-connection to 3 separate classes (Res, Pre, Post) instead of one class because
# they are all initialized differently and there is 3 different forward passes for each of them
#
# NOTE: For reference, per the mHC paper p.10:
# - inputs (xl and xl_norm) are in bf16
# - H_res, H_pre, H_post, scaling factors and static mapping (biases) are stored and computed in fp32.
# - dynamic mappings (phi_res, phi_pre, phi_post) are in tf32 (Nvidia TensorFloat-32)

import torch
import torch.nn as nn


class HyperConnectionRes(nn.Module):
    """
    Classic Hyper-connection for residual stream as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 3 and 5): https://arxiv.org/abs/2512.24880
    - Hyper-Connections (HC) paper: https://arxiv.org/abs/2409.19606

    This class is basically returning the residual hyperconnection output of H_res @ x described in eq 3 of the mHC
    paper.

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The activation function class, default is Tanh per the mHC paper
        device (torch.device):
        dtype (torch.dtype):

    Returns:
        x: The mixed residual streams, shape: (b, seq_len, exps_rate, emb_dim)
    """

    def __init__(
        self,
        emb_dim,
        expansion_rate=4,
        add_static_mapping=True,
        activation_cls=nn.Tanh,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (phi_res): project to n streams
        self.linear = nn.Linear(emb_dim, expansion_rate, bias=False, device=device, dtype=dtype)
        # The HC paper init dynamic mapping weights for phi_res as 0 (HC paper p.4 section 2.3) but Pytorch nn.linear
        # is doing Kaiming init by default so we need to zero init
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_res) are initialized as the identity matrix (HC paper), since dynamic mapping weights
        # are initialized to 0, overall, this makes the hyper-connection start with an untouched stream (like a classic
        # residual connection): (0 + I) @ xl = xl
        self.bias = (
            nn.Parameter(torch.eye(expansion_rate, device=device, dtype=dtype)) if add_static_mapping else None
        )  # shape (exps_rate, exps_rate)

    # TODO mention diff with DeepSeek: expansion stream are flattened, (n, emb_dim) -> (n*emb_dim) no need for transpose
    def residual_matrix(self, x_norm):
        """
        Generates the residual mapping/matrix H_res
        This is our small "HyperNet" where we generate on the fly the weights H_res to mix with the residual stream

        Args:
            x_norm: The n flattened streams, normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, 1, n*emb_dim)

        Returns:
            The residual mapping/matrix H_res, shape: (b, seq_len, exps_rate, exps_rate)
        """
        x = self.linear(x_norm)  # apply dynamic mapping
        # Pytorch nn.linear is doing Wx (as XW^T) but we want WX^T (eq 5), therefore we need to transpose the linear
        # output as (XW^T)^T = (W^T)^T X^T = WX^T
        x = x.mT

        # activate and scale
        x = self.activation(x) * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        return x

    def forward(self, x, x_norm):
        """
        Apply/mix the residual mapping/matrix H_res to the residual stream

        Args:
            x: The n streams input, shape: (b, seq_len, exps_rate, emb_dim)
            x_norm: The n streams normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The n mixed streams after applying the residual mapping H_res, shape: (b, seq_len, exps_rate, emb_dim)
        """
        x = self.residual_matrix(x_norm) @ x

        return x


class MCHyperConnectionPre(nn.Module):
    """
    TODO changing comments
    Manifold-Constrained Hyper-connection for the entry of the residual branch/pre-transformer block, as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 3 and 5): https://arxiv.org/abs/2512.24880

    This class is basically downprojecting the n expanded streams to a single stream, in order to pass into the
    transformer block, returning the output of H_pre @ x, described in eq 3 of the mHC paper.

    Since the trf block has to be a single stream (to match traditional LLM implementation and not increase the time
    complexity), H_pre weights serve to scale how much each of the n expanded streams contributes to this single merged
    stream.

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The activation function class, default is Tanh per the mHC paper
        device (torch.device):
        dtype (torch.dtype):

    Returns:
        x: The single stream resulting of the aggregated expanded streams, after applying the pre mapping H_pre, ready
        for the trf block, shape: (b, seq_len, emb_dim)
    """

    def __init__(
        self,
        emb_dim,
        expansion_rate=4,
        add_static_mapping=True,
        activation_cls=nn.Sigmoid,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (phi_pre): downproject the emb dim to a scalar:
        # This determines how much the trf block output contributes to each of the n expanded streams
        self.linear = nn.Linear(emb_dim * expansion_rate, expansion_rate, bias=False, device=device, dtype=dtype)
        # Same init for all dynamic mapping weights as 0 (HC paper p.4 section 2.3)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_pre) initialization:
        # - mHC paper (table 1 p.6): uniform averaged weights 1/n, mixed
        # TODO might need to rescale because of the sigmoid
        self.bias = (
            nn.Parameter(torch.ones(expansion_rate, device=device, dtype=dtype) / expansion_rate)
            if add_static_mapping
            else None
        )  # shape (exps_rate,)

    def pre_mapping_matrix(self, x_norm):
        """
        Generates the pre trf block mapping/matrix H_pre
        This is our small "HyperNet" where we generate on the fly the weights H_pre. A weight for each of the n expanded
        streams that will determine/scale their contribution to the aggregated single stream for the trf block.

        Args:
            x_norm: The n flattened streams, normalized input (pre-trf block), used to generate H_pre,
                    shape: (b, seq_len, 1, exps_rate*emb_dim)

        Returns:
            The pre mapping/matrix H_pre as a row vector, shape: (b, seq_len, 1, exps_rate)
        """
        # shape (b, seq_len, 1, exps_rate*emb_dim) → (b, seq_len, exps_rate)
        x = self.linear(x_norm) * self.factor  # apply dynamic mapping and factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        x = self.activation(x)  # constrain with sigmoid

        return x.unsqueeze(-2)

    def forward(self, x, x_norm):
        """
        Aggregate/collapse the n expanded streams to a single stream with the pre mapping matrix H_pre weights

        This ends up being a weighted sum of the n streams using the dynamically generated H_pre weights.

        Args:
            x: The n streams input, shape: (b, seq_len, exps_rate, emb_dim)
            x_norm: The n streams normalized input (pre-trf block), used to generate H_pre,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The aggregated single stream ready for the trf block, shape: (b, seq_len, emb_dim)
        """

        x = self.pre_mapping_matrix(x_norm) @ x
        x = x.squeeze(-2)  # shape (b, seq_len, 1, emb_dim) → (b, seq_len, emb_dim)

        return x


class HyperConnectionPost(nn.Module):
    """
    Classic Hyper-connection for the exit of the residual branch/post-transformer block, as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 3 and 5): https://arxiv.org/abs/2512.24880
    - Hyper-Connections (HC) paper: https://arxiv.org/abs/2409.19606

    This class is basically broadcasting the single stream output of the trf block back to the n expanded streams
    weighted by their H_post weights. This is done in order to mix/add it with the other n streams from the main
    branch (output from H_res @ x).
    It returns the output of (H_post^T @ output trf block), described in eq 3 of the mHC paper.

    Since we are operating with n streams and the trf block had to be a single stream (to match traditional LLM
    implementation and not increase the time complexity), H_post weights serve to scale how much each broadcasted trf
    block output contributes to each of the n expanded streams.
    ex: stream[i] = H_post_weight[i] * trf_output

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The activation function class, default is Tanh per the mHC paper
        device (torch.device):
        dtype (torch.dtype):

    Returns:
        x: The n post-trf block scaled streams, after applying the post mapping H_post,
        shape: (b, seq_len, exps_rate, emb_dim)
    """

    def __init__(
        self,
        emb_dim,
        expansion_rate=4,
        add_static_mapping=True,
        activation_cls=nn.Tanh,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (phi_post), same logic as phi_pre:
        # This determines how much the trf block output contributes to each of the n expanded streams
        self.linear = nn.Linear(emb_dim, 1, bias=False, device=device, dtype=dtype)
        # Same init for all dynamic mapping weights as 0 (HC paper p.4 section 2.3)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_post) initialization:
        # uniform weights of 1s
        self.bias = (
            nn.Parameter(torch.ones(expansion_rate, device=device, dtype=dtype)) if add_static_mapping else None
        )  # shape (exps_rate,)

    def post_mapping_matrix(self, x_norm):
        """
        Generates the post mapping/matrix H_post
        This is our small "HyperNet" where we generate on the fly the weights H_post. A weight for each of the n
        expanded streams that will scale the broadcasted trf block output contribution for each of the n
        expanded streams.

        Args:
            x_norm: The n streams normalized input (post-trf block), used to generate H_post,
                    shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The transposed post mapping/matrix H_post^T as a column vector, shape: (b, seq_len, exps_rate, 1)
        """
        # shape (b, seq_len, exps_rate, emb_dim) → (b, seq_len, exps_rate, 1) → (b, seq_len, exps_rate)
        x = self.linear(x_norm).squeeze(-1)  # apply dynamic mapping
        x = self.activation(x) * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        return x.unsqueeze(-1)  # unsqueeze serve as transpose here H_post^T making it a column vector

    def forward(self, x, x_norm):
        """
        Broadcast the trf block output back to the n expanded streams and scaled with the post mapping matrix H_post
        weights.

        Args:
            x: The single stream output of the trf block, shape: (b, seq_len, emb_dim)
            x_norm: The n streams normalized input (post-trf block), used to generate H_post,
                    shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The n mixed streams, shape: (b, seq_len, exps_rate, emb_dim)
        """

        x = self.post_mapping_matrix(x_norm) @ x.unsqueeze(-2)
        return x


# quick inline test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)
    device = config.auto_device
    dtype = torch.bfloat16

    test_x = torch.randn(1, 10, 4, 128, device=device, dtype=dtype)
    test_x_norm = torch.nn.functional.normalize(test_x.view(1, 10, -1), dim=-1)
    # test_hc_res = HyperConnectionRes(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    test_hc_pre = MCHyperConnectionPre(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    # test_hc_post = HyperConnectionPost(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    # test_output_res = test_hc_res(test_x, test_x_norm)
    test_output_pre = test_hc_pre(test_x, test_x_norm)
    # test_output_post = test_hc_post(test_output_pre, test_x_norm)
    # print(test_output_res.shape)
    print(test_output_pre.shape)
    # print(test_output_post.shape)
