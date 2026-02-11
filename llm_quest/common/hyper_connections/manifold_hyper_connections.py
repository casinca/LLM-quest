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
# - H_res, H_pre, H_post, scaling factors and static mapping (biases) are stored and computed in fp32. TODO
# - dynamic mappings (phi_res, phi_pre, phi_post) are in tf32 (Nvidia TensorFloat-32)

import math

import torch
import torch.nn as nn

from llm_quest.utils import BirkhoffvonNeumann, SinkhornKnopp


class MCHyperConnectionRes(nn.Module):
    """
    Manifold-Constrained Hyper-connection for residual stream as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 7 and 8): https://arxiv.org/abs/2512.24880

    This class is basically returning the residual constrained hyperconnection output of H_res @ x described in eq 3 of
    the mHC paper.

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The SinkhornKnopp class instance to make the residual matrix doubly stochastic
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
        constraint_cls=SinkhornKnopp,
        sk_max_iter=20,
        sk_epsilon=1e-6,
        sk_iter_check=3,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.sinkhorn_knopp = constraint_cls(max_iter=sk_max_iter, epsilon=sk_epsilon, iter_check=sk_iter_check)

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (phi_res): project to n² streams
        self.linear = nn.Linear(emb_dim * expansion_rate, expansion_rate**2, bias=False, device=device, dtype=dtype)
        # The HC paper init dynamic mapping weights for phi_res as 0 (HC paper p.4 section 2.3) but Pytorch nn.linear
        # is doing Kaiming init by default so we need to zero init
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_res) are initialized as the identity matrix (mHC & HC paper), since dynamic mapping
        # weights are initialized to 0, overall, this makes the hyper-connection start with an untouched stream (like a
        # classic residual connection): (0 + I) @ xl = xl
        if add_static_mapping:
            # since we use exponential to make values > 0 for SK, if bias is left as default init as I, the
            # identity property will be lost after torch.exp().
            # Therefore we re-init bias so that it approximates I after torch.exp()
            # values: 0 for diags / -8 for the rest (why -8? inspired from mHC-lite paper init, exp^(-8) ~ 0)
            init_bias = torch.eye(expansion_rate, device=device, dtype=dtype) * 8 - 8
            self.bias = nn.Parameter(init_bias)  # shape (exps_rate, exps_rate)
        else:
            self.bias = None

    def residual_matrix(self, x_norm):
        """
        Generates the residual mapping/matrix H_res
        This is our small "HyperNet" where we generate on the fly the weights H_res to mix with the residual stream

        Args:
            x_norm: The n flattened streams, normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The residual mapping/matrix H_res, shape: (b, seq_len, exps_rate, exps_rate)
        """
        b, seq_len, _ = x_norm.shape
        # shape (b, seq_len, exps_rate*emb_dim) → (b, seq_len, exps_rate^2) → (b, seq_len, exps_rate, exps_rate)
        x = self.linear(x_norm).view(b, seq_len, self.expansion_rate, self.expansion_rate)  # apply dynamic mapping

        x = x * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        x = torch.exp(x)  # map to positive range for SK
        x = self.sinkhorn_knopp(x)

        return x

    def forward(self, x, x_norm):
        """
        Apply/mix the residual mapping/matrix H_res to the residual stream

        Args:
            x: The n streams input, shape: (b, seq_len, exps_rate, emb_dim)
            x_norm: The n streams normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The n mixed streams after applying the residual mapping H_res, shape: (b, seq_len, exps_rate, emb_dim)
        """
        x = self.residual_matrix(x_norm) @ x

        return x


class MHCLiteRes(nn.Module):
    """
    This is the residual mapping from the mHC-lite paper: https://arxiv.org/abs/2601.05732

    Main changes vs DeepSeek mHC:
    - Compute doubly stochastic matrix H_res directly from convex combination (Birkhoff-von Neumann theorem)
    - dynamic mapping is projeted to n! streams instead of n² streams
    - static mapping are init differently to take into account that we deal with a vector of size n! instead of a matrix
    of size n²

    This class is basically returning the residual constrained hyperconnection output of H_res @ x described in eq 3 of
    the mHC paper.

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        constraint_cls (nn.Module): The constraint class instance to make the residual matrix doubly stochastic
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
        constraint_cls=BirkhoffvonNeumann,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.num_permuts = math.factorial(expansion_rate)
        self.bvn = constraint_cls(expansion_rate=self.expansion_rate).to(device)

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (W_res in mHC-lite paper): project to n! streams
        self.linear = nn.Linear(emb_dim * expansion_rate, self.num_permuts, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_res) are scalars in a vector, not a matrix (1 weight for each of the n! permutations)
        # Per the mHC-lite paper, p.5:
        # They are all init to -8 for getting a small value after softmax, except the identity permutation
        # (ex: if n=4, the identity permutation is (0,1,2,3)) which is set to 0.
        if add_static_mapping:
            self.bias = nn.Parameter(torch.full(size=(self.num_permuts,), fill_value=-8, device=device, dtype=dtype))
            with torch.no_grad():
                self.bias[self.bvn.identity_permut_index] = 0  # set the identity permutation to 0
        else:
            self.bias = None

    def residual_matrix(self, x_norm):
        """
        Generates the residual mapping/matrix H_res
        This is our small "HyperNet" where we generate on the fly the weights H_res to mix with the residual stream

        Args:
            x_norm: The n flattened streams, normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The residual mapping/matrix H_res, shape: (b, seq_len, exps_rate, exps_rate)
        """
        b, seq_len, _ = x_norm.shape
        # shape (b, seq_len, exps_rate*emb_dim) → (b, seq_len, n!)
        x = self.linear(x_norm)  # apply dynamic mapping

        x = x * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        x = torch.softmax(x, dim=-1)  # logits to `weight_a` (coeffs/weights to positive range for BVN)
        x = self.bvn(x)  # apply our convex combination/weighted average from the BVN theorem

        return x

    def forward(self, x, x_norm):
        """
        Apply/mix the residual mapping/matrix H_res to the residual stream

        Args:
            x: The n streams input, shape: (b, seq_len, exps_rate, emb_dim)
            x_norm: The n streams normalized input (pre-trf block), used to generate H_res,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The n mixed streams after applying the residual mapping H_res, shape: (b, seq_len, exps_rate, emb_dim)
        """
        x = self.residual_matrix(x_norm) @ x

        return x


class MCHyperConnectionPre(nn.Module):
    """
    Manifold-Constrained Hyper-connection for the entry of the residual branch/pre-transformer block, as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 7 and 8): https://arxiv.org/abs/2512.24880

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
        activation_cls (nn.Module): The activation function class, use a sigmoid for constraining (non-negative)
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

        # dynamic mapping (phi_pre): downproject the flattened/mixed streams to exps_rate
        # (will get a single weight for each of the n expanded streams)
        self.linear = nn.Linear(emb_dim * expansion_rate, expansion_rate, bias=False, device=device, dtype=dtype)
        # Same init for all dynamic mapping weights as 0 (HC paper p.4 section 2.3)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_pre) initialization:
        # - mHC paper (table 1 p.6): uniform averaged weights 1/n, mixed
        # Here, in order to make sigmoid(H_pre_tilde) = 1/n, with H_pre_tilde = bias (since phi_pre is 0 init)
        # we need to solve for sigmoid(b) = 1/n, which gives b = -ln(n-1)
        # edge case: can't do ln(0)
        init_val = -math.log(expansion_rate - 1) if expansion_rate > 1 else 10  # sigmoid(10) ~ 1
        self.bias = (
            nn.Parameter(torch.full(size=(expansion_rate,), fill_value=init_val, device=device, dtype=dtype))
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
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The pre mapping/matrix H_pre as a row vector, shape: (b, seq_len, 1, exps_rate)
        """
        # shape (b, seq_len, exps_rate*emb_dim) → (b, seq_len, exps_rate)
        x = self.linear(x_norm) * self.factor  # apply dynamic mapping and factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        x = self.activation(x)  # constrain H_pre_tilde with sigmoid

        return x.unsqueeze(-2)

    def forward(self, x, x_norm):
        """
        Aggregate/collapse the n expanded streams to a single stream with the pre mapping matrix H_pre weights

        This ends up being a weighted sum of the n streams using the dynamically generated H_pre weights.

        Args:
            x: The n streams input, shape: (b, seq_len, exps_rate, emb_dim)
            x_norm: The n flattened streams normalized input (pre-trf block), used to generate H_pre,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The aggregated single stream ready for the trf block, shape: (b, seq_len, emb_dim)
        """

        x = self.pre_mapping_matrix(x_norm) @ x
        x = x.squeeze(-2)  # shape (b, seq_len, 1, emb_dim) → (b, seq_len, emb_dim)

        return x


class MCHyperConnectionPost(nn.Module):
    """
    Manifold-Constrained Hyper-connection for the exit of the residual branch/post-transformer block, as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 7 and 8): https://arxiv.org/abs/2512.24880

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
        activation_cls (nn.Module): The activation function class, use a sigmoid for constraining (non-negative)
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
        activation_cls=nn.Sigmoid,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (phi_post), same logic as phi_pre:
        # This determines how much the trf block output contributes to each of the n expanded streams
        self.linear = nn.Linear(emb_dim * expansion_rate, expansion_rate, bias=False, device=device, dtype=dtype)
        # Same init for all dynamic mapping weights as 0 (HC paper p.4 section 2.3)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_post) initialization:
        # uniform weights of 1s
        # Here, in order to make sigmoid(H_post_tilde) = 1, with H_post_tilde = bias (since phi_post is 0 init)
        # we need to solve for 2*sigmoid(b) = 1, which gives b = -ln(1) = 0
        self.bias = (
            nn.Parameter(torch.zeros(expansion_rate, device=device, dtype=dtype)) if add_static_mapping else None
        )  # shape (exps_rate,)

    def post_mapping_matrix(self, x_norm):
        """
        Generates the post mapping/matrix H_post
        This is our small "HyperNet" where we generate on the fly the weights H_post. A weight for each of the n
        expanded streams that will scale the broadcasted trf block output contribution for each of the n
        expanded streams.

        Args:
            x_norm: The n flattened streams normalized input (post-trf block), used to generate H_post,
                    shape: (b, seq_len, exps_rate*emb_dim)

        Returns:
            The transposed post mapping/matrix H_post^T as a column vector, shape: (b, seq_len, exps_rate, 1)
        """
        # shape (b, seq_len, exps_rate*emb_dim) → (b, seq_len, exps_rate)
        x = self.linear(x_norm) * self.factor  # apply dynamic mapping and factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        x = self.activation(x) * 2  # constrain H_post_tilde with sigmoid and rescale

        return x.unsqueeze(-1)  # unsqueeze serve as transpose here H_post^T making it a column vector

    def forward(self, x, x_norm):
        """
        Broadcast the trf block output back to the n expanded streams and scaled with the post mapping matrix H_post
        weights.

        Args:
            x: The single stream output of the trf block, shape: (b, seq_len, emb_dim)
            x_norm: The n flattened streams normalized input (post-trf block), used to generate H_post,
                    shape: (b, seq_len, exps_rate*emb_dim)

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

    test_mhc_res = MCHyperConnectionRes(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    test_mhc_pre = MCHyperConnectionPre(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    test_mhc_post = MCHyperConnectionPost(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)

    test_mhc_lite_res = MHCLiteRes(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)

    test_output_res = test_mhc_res(test_x, test_x_norm)
    test_output_pre = test_mhc_pre(test_x, test_x_norm)
    test_output_post = test_mhc_post(test_output_pre, test_x_norm)
    test_output_lite_res = test_mhc_lite_res(test_x, test_x_norm)
    print("res: ", test_output_res.shape)
    print("pre: ", test_output_pre.shape)
    print("post: ", test_output_post.shape)
    print("lite_res: ", test_output_lite_res.shape)

    print("\n\n# Check if the residual mixing matrix is doubly stochastic with mHC SK")
    H_res = test_mhc_res.residual_matrix(test_x_norm)  # (b, seq_len, exp_rate, exp_rate)
    row_sums = H_res.sum(dim=-1)  # sum over columns, shape: (b, seq_len, exp_rate)
    col_sums = H_res.sum(dim=-2)  # sum over rows,    shape: (b, seq_len, exp_rate)
    print("Row sums (should be ~1):", row_sums)
    print("Col sums (should be ~1):", col_sums)

    print("\n\n# Check if the residual mixing matrix is doubly stochastic with mHC Lite BVN")
    H_res_lite = test_mhc_lite_res.residual_matrix(test_x_norm)  # (b, seq_len, exp_rate, exp_rate)
    row_sums_lite = H_res_lite.sum(dim=-1)  # sum over columns, shape: (b, seq_len, exp_rate)
    col_sums_lite = H_res_lite.sum(dim=-2)  # sum over rows,    shape: (b, seq_len, exp_rate)
    print("Row sums (should be ~1):", row_sums_lite)
    print("Col sums (should be ~1):", col_sums_lite)
