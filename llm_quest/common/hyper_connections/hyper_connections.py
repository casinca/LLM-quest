import torch
import torch.nn as nn

# separating each hyper-connection to 3 separate classes (Res, Pre, Post) instead of one class because
# they are all initialized differently and there is 3 different forward passes for each of them

# TODO the norms are duplicated, it should be a single norm for all 3 HC per layer, either we do it in the model
# forward or we wrap into a HyperConnection class
# TODO the input doc says n untouched/identity/residual streams input but it's only for the first layer, rest are modified
# streams for previous layers so not identity anymore


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
        norm_cls (nn.Module): The normalization function class, default is RMSNorm per the mHC paper
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
        norm_cls=nn.RMSNorm,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.norm = norm_cls(emb_dim, device=device, dtype=dtype)  # TODO recheck dtype for norm
        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (theta_res): project to n streams
        self.linear = nn.Linear(emb_dim, expansion_rate, bias=False, device=device, dtype=dtype)
        # The HC paper init dynamic mapping weights for theta_res as 0 (HC paper p.4 section 2.3) but Pytorch nn.linear
        # is doing Kaiming init by default so we need to zero init
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_res) are initialized as the identity matrix (HC paper), since dynamic mapping weights
        # are initialized to 0, overall, this makes the hyper-connection start with an untouched stream (like a classic
        # residual connection): (0 + I) @ xl = xl
        self.bias = (
            nn.Parameter(torch.eye(expansion_rate, device=device, dtype=dtype)) if add_static_mapping else None
        )  # shape (exps_rate, exps_rate)

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


class HyperConnectionPre(nn.Module):
    """
    Classic Hyper-connection for the entry of the residual branch/pre-transformer block, as depicted in:
    - DeepSeek mHC: Manifold-Constrained Hyper-Connections paper (eq 3 and 5): https://arxiv.org/abs/2512.24880
    - Hyper-Connections (HC) paper: https://arxiv.org/abs/2409.19606

    This class is basically downprojecting the n expanded streams to a single stream, in order to pass into the
    transformer block, returning the output of H_pre @ x, described in eq 3 of the mHC paper.

    Args:
        emb_dim (int): The dimension of the embeddings
        expansion_rate (int): The number of expanded streams, ("n" in the paper), can be seen as the width of the
                            residual stream
        add_static_mapping (bool): Whether to add static mappings, ie adding biases (b_res in mHC the paper)
        activation_cls (nn.Module): The activation function class, default is Tanh per the mHC paper
        norm_cls (nn.Module): The normalization function class, default is RMSNorm per the mHC paper
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
        activation_cls=nn.Tanh,
        norm_cls=nn.RMSNorm,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.norm = norm_cls(emb_dim, device=device, dtype=dtype)  # TODO recheck dtype for norm
        self.activation = activation_cls()

        # scalar learnable gating factor, (alpha in the mHC paper, table 5 hparam init=0.01)
        self.factor = nn.Parameter(torch.tensor([0.01], device=device, dtype=dtype))

        # dynamic mapping (theta_pre): downproject to a single stream
        self.linear = nn.Linear(emb_dim, 1, bias=False, device=device, dtype=dtype)
        # Same init for all dynamic mapping weights as 0 (HC paper p.4 section 2.3)
        nn.init.zeros_(self.linear.weight)

        # static mapping (biases b_pre) initialization:
        # There is a disagreement between the HC paper (p.4 eq 14) and mHC paper (table 1 p.6)
        # - HC paper: One hot like, selecting 1 stream out of n depending on the trf layer, not mixed
        #           hence the use of modulo which cycles through the streams with the layer index (layer_idx % n)
        # - mHC paper: uniform averaged weights 1/n, mixed
        # We follow DeepSeek
        self.bias = (
            nn.Parameter(torch.ones(expansion_rate, device=device, dtype=dtype) / expansion_rate)
            if add_static_mapping
            else None
        )  # shape (exps_rate,)

    def pre_mapping_matrix(self, x):
        """
        Generates the pre trf block mapping/matrix H_pre
        This is our small "HyperNet" where we generate on the fly the weights H_pre. A weight for each of the n expanded
        streams that will determine/scale their contribution to the aggregated single stream for the trf block.

        Args:
            x: The n untouched/identity/residual streams input, shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The pre mapping/matrix H_pre as a row vector, shape: (b, seq_len, 1, exps_rate)
        """
        x = self.norm(x)
        # shape (b, seq_len, exps_rate, emb_dim) → (b, seq_len, exps_rate, 1) → (b, seq_len, exps_rate)
        x = self.linear(x).squeeze(-1)  # apply dynamic mapping
        x = self.activation(x) * self.factor

        if self.bias is not None:  # add static mapping if enabled
            x += self.bias

        return x.unsqueeze(-2)

    def forward(self, x):
        """
        Aggregate/collapse the n expanded streams to a single stream via the pre mapping matrix H_pre

        This ends up being a weighted sum of the n streams using the dynamically generated H_pre weights.

        Args:
            x: The n untouched/identity/residual streams input, shape: (b, seq_len, exps_rate, emb_dim)

        Returns:
            The aggregated single stream ready for the trf block, shape: (b, seq_len, emb_dim)
        """

        x = self.pre_mapping_matrix(x) @ x
        x = x.squeeze(-2)  # shape (b, seq_len, 1, emb_dim) → (b, seq_len, emb_dim)

        return x


# quick inline test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)
    device = config.auto_device
    dtype = torch.bfloat16

    test_x = torch.randn(1, 10, 4, 128, device=device, dtype=dtype)
    test_hc_res = HyperConnectionRes(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    test_hc_pre = HyperConnectionPre(emb_dim=128, expansion_rate=4, device=device, dtype=dtype)
    test_output_res = test_hc_res(test_x)
    test_output_pre = test_hc_pre(test_x)
    print(test_output_res.shape)
    print(test_output_pre.shape)
