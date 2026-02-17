import torch
import torch.nn as nn


# NOTE: Paper does W' = Wx + BAx, in pytorch we do W' = xW^T + x @ A^T @ B^T for x coming from the left
# Here we store A and B already transposed, so xW^T + xAB
class LoRALinearLayer(nn.Module):
    """
    LoRA changes the default linear layers (paper mentions Attention weights, but any learnable weights) to add 2
    lower rank matrices AB for learning (instead of directly updating W of shape (d_in, d_out or d, k) with a ∆W of same
    size).
    The update isn't done to W (frozen) but separately learned through AB.

    The intuition is that we can replace a full weight update matrix ∆W by approximating it with 2 low-rank matrices.
    That's because fine-tuning updates are generally close to low-rank, hence we can explicitly learn these updates to
    the original model in their factorized form.

    ie, our new forward pass becomes xW^T → xW^T + xAB where W is frozen and gradient flows through AB instead of ∆W.
    with W of shape (d, k) and A, B of respective shapes (d, r) and (r, k)
    With r a rank based on num of params and weights (r ∈ {1, 2, 4, ..., 2^n} a good starting point per the paper)

    The fact that gradient updates with AB are decoupled from the weights, gives us the possibility to store and
    reuse any AB for another downstream task at no additional inference latency (vs re-finetuning).
    ex: xW^T - xAB to get back default pretrained weights and xW^T + xA'B' to use another set of updated weights.

    paper: https://arxiv.org/abs/2106.09685

    Args:
        trained_linear_layer (nn.Linear): The pretrained linear layer to wrap (will be frozen).
        r (int): Rank of the low-rank matrices (r << d, k).
        alpha (float): Scaling factor for the update.

    Returns:
        torch.Tensor: The output tensor after the forward pass: xW^T + α/r * xAB.
    """

    def __init__(self, trained_linear_layer, r, alpha):
        super().__init__()
        self.linear = trained_linear_layer
        ref = self.linear.weight  # used for retrieving og dtype and device
        # in case (but should already be done upstream)
        for param in self.linear.parameters():
            param.requires_grad = False

        d = self.linear.in_features
        k = self.linear.out_features

        # zero init for B (nn.param flags the tensor as a learnable param)
        self.B = nn.Parameter(torch.zeros(r, k).to(ref))

        # random Gaussian init for A (per the paper)
        # LoRA code from MSFT and @rasbt's alt: nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.A = nn.Parameter(torch.empty(d, r).to(ref))
        nn.init.normal_(self.A, mean=0.0, std=0.02)  # small std commonly used

        # AB is scaled by α/r, where α is a constant scaling factor: α/r * xAB
        # (α is basically the same as the 'α' factor in Adam to tune the lr)
        self.scaler = alpha / r

    def forward(self, x):
        # return standard linear layer + our low-rank parametrized update matrices α/r * xAB (LoRA)
        return self.linear(x) + self.scaler * (x @ self.A @ self.B)


# for completeness (will only change Attention Q,K,V and output layer for instruct training like the paper)
# (iterating through model.modules() is faster and doesn't need explicit dfs but more verbose than named.children())
def replace_with_lora(model, rank, alpha, lora_class=LoRALinearLayer):
    """
    This function iterates through all modules in the given model. When it finds a nn.Linear layer,
    it replaces it with a LoRALinearLayer that wraps the existing (pretrained) linear and freezes it,
    with the specified rank and alpha for the LoRA update.
    For modules that are not nn.Linear, it recursively calls itself to check and replace
    any nn.Linear layers within those submodules.

    Args:
        model (nn.Module): The model in which to replace layers (must have pretrained weights loaded).
        rank (int): The rank 'r' for the low-rank update.
        alpha (float): The scaling factor for the low-rank update.
        lora_class: Class with signature (trained_linear_layer, r, alpha), e.g. LoRALinearLayer or LoRAXSLinearLayer.
    """
    # iterating through modules, ex: ("linear1", nn.Linear(10, 5)), ("relu", nn.ReLU())
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, lora_class(module, rank, alpha))
        else:
            # recursively check & replace in submodules
            replace_with_lora(module, rank, alpha, lora_class=lora_class)


class LoRAXSLinearLayer(nn.Module):
    """
    LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters.
    paper: https://arxiv.org/abs/2405.17604

    Instead of learning 2 low ranks matrices A and B (as in LoRA), LoRA-XS learns a single (r, r) matrix R.
    LoRA: W'= W + BA

    LoRA-XS: W'= W + UΣRV^T
        with UΣ = A and V^T = B

        U shape (d, r), Σ (r, r), V^T (r, k) are computed from the SVD of W, and are frozen.

        More specifically we take U_r, V_r^T and Σ_r from the truncated SVD of W.
        Which means the top r vectors from U and V^T and the top r singular values from Σ.

        The computation is done thanks to `torch.linalg.svd()` which return values in descending order.

    Overall LoRA-XS, reduces the number of trainable parameters from 2 matrices BA, 2*d*r (as in LoRA) to a single
    r² one (r << d, k), making it independent of the model's hidden dimensions.

    Args:
        trained_linear_layer (nn.Linear): The trained linear layer to be replaced with LoRA-XS (linear+LoRA-XS update)
        r (int): Rank of the low-rank adaptation (r << d, k)
        alpha (float): Scaling factor for the update (paper uses α = r for instruction tuning, 16 for GLUE benchmark)

    Returns:
        torch.Tensor: The output tensor after the forward pass: xW^T + α/r * xARB.
    """

    def __init__(self, trained_linear_layer, r, alpha):
        super().__init__()

        # Store the frozen trained linear layer
        self.linear = trained_linear_layer
        ref = self.linear.weight  # used for retrieving og dtype and device
        # in case
        for param in self.linear.parameters():
            param.requires_grad = False

        # Compute truncated SVD of the trained weight/matrix
        # W = U @ Σ @ V^T, with W as shape (d, k) so we transpose because nn.Linear stores weight as (k, d)
        with torch.no_grad():
            # can't use "Σ" symbol, so "S" for sigmas. Also doing SVD in fp32
            U, S, Vt = torch.linalg.svd(self.linear.weight.data.float().T, full_matrices=False)

            # truncating: top r singular vectors and values and cast to the same dtype and device of the linear layer
            # U_r shape (d, r), Vt_r shape (r, k)
            # S_r shape (r,) PyTorch for efficiency returns a vector, instead of a diag matrix with 0s
            U_r, S_r, Vt_r = [t.to(ref) for t in [U[:, :r], S[:r], Vt[:r, :]]]

            # Unlike LoRA, AB are frozen, not trained (registered as buffers), only R is trained
            self.register_buffer("A", U_r @ torch.diag(S_r))  # A = U_r * Σ_r (shape: d, r)
            self.register_buffer("B", Vt_r)  # B = V_r^T (shape: r, k)

        # Normal init N(0, σ²) with σ = 1e-5 per the paper
        self.R = nn.Parameter(torch.empty(r, r).to(ref))
        nn.init.normal_(self.R, mean=0.0, std=1e-5)

        # paper mentions p.12: α = r for instruction tuning (so scaler = 1), α = 16 for GLUE benchmark
        self.scaler = alpha / r

    def forward(self, x):
        """Forward pass: h = xW^T + α/r * xARB"""
        # linear layer + LoRA-XS update
        return self.linear(x) + self.scaler * (x @ self.A @ self.R @ self.B)


class TinyLoRALinearLayer(nn.Module):
    """
    TinyLoRA: Learning to Reason in 13 Parameters
    paper: https://arxiv.org/abs/2602.04118

    Instead of learning a single (r, r) matrix R (like LoRA-XS), TinyLoRA learns a single vector v (u,) that is used to
    generate the matrix R (the same matrix from LoRA-XS).

    More specifically, R is generated from the sum of scalar matrix products between "u" scalars and "u" fixed matrices
    P_i (i = 1,..., u) with P_i (r, r)

    What we end up doing is creating R from a sum of matrices P_i weighted by the learned scalars v_i.
    ie: R = sum(v_i * P_i) with i=1,...,u

    The rest of the logic is the same as in LoRA-XS. Ie, AB from the truncated SVD of W.

    LoRA-XS: W'= W + UΣRV^T
        with UΣ = A and V^T = B

    TinyLoRA: W'= W + UΣ sum(v_i * P_i) V^T
        with sum(v_i * P_i) = R and i=1,...,u

    TinyLoRA also proposes tying the vector v across modules and layers.
    Therefore with a 1-dim vector (u=1) and global weight tying, we can end up with as low as 1 parameter to train.

    NOTE: we compute R as a matmul (v @ P_flat) just like in mHC-lite with BVN/convex combination and not directly as a
    weighted sum (torch.sum(v * P, dim=0)) as in the paper, for efficiency.
    Although r being usually small, unlikely that it'll matter much here...

    Args:
        trained_linear_layer (nn.Linear): The trained linear layer to be replaced with TinyLoRA (linear+TinyLoRA update)
        r (int): Rank of the low-rank adaptation (r << d, k)
        alpha (float): Scaling factor for the update
        num_trainable_params (int): Number of trainable parameters, this is "u" in the paper, dim of the vector v
        (default is 13 to match the paper)
        shared_v: The vector v to share across modules and layers, if None, a new vector v will be created

    Returns:
        torch.Tensor: The output tensor after the forward pass: xW^T + α/r * xARB.
                    with R = sum(v_i * P_i)
    """

    def __init__(self, trained_linear_layer, r, alpha, num_trainable_params=13, shared_v=None):
        super().__init__()
        self.rank = r  # this is just for the forward to reshape R
        self.linear = trained_linear_layer
        ref = self.linear.weight  # used for retrieving og dtype and device
        # in case
        for param in self.linear.parameters():
            param.requires_grad = False

        # Compute truncated SVD of the trained weight/matrix
        # W = U @ Σ @ V^T, with W as shape (d, k) so we transpose because nn.Linear stores weight as (k, d)
        with torch.no_grad():
            # can't use "Σ" symbol, so "S" for sigmas. Also doing SVD in fp32
            U, S, Vt = torch.linalg.svd(self.linear.weight.data.float().T, full_matrices=False)

            # truncating: top r singular vectors and values
            # U_r shape (d, r), Vt_r shape (r, k)
            # S_r shape (r,) PyTorch for efficiency returns a vector, instead of a diag matrix with 0s
            U_r, S_r, Vt_r = [t.to(ref) for t in [U[:, :r], S[:r], Vt[:r, :]]]

            # Unlike LoRA, AB are frozen, not trained (registered as buffers)
            self.register_buffer("A", U_r @ torch.diag(S_r))  # A = U_r * Σ_r (shape: d, r)
            self.register_buffer("B", Vt_r)  # B = V_r^T (shape: r, k)

            # they just mention random fixed matrices, so N(0, 1) should look good
            # not (u, r, r) shape as in the paper, here (u, r²) to do the weighted sum as a matmul
            self.register_buffer("P", torch.randn(num_trainable_params, r * r).to(ref))

        if shared_v is not None:
            if shared_v.shape[0] != num_trainable_params:
                raise ValueError(f"shared_v is {shared_v.shape} and num params {num_trainable_params} don't match")
            self.v = shared_v
        else:
            # no mention of init, but we want to start training at 0 (ARB=0) and since P is N random, v should be 0
            self.v = nn.Parameter(torch.zeros(num_trainable_params).to(ref))

        self.scaler = alpha / r

    def forward(self, x):
        """Forward pass: h = xW^T + α/r * xARB with R = sum(v_i * P_i)"""
        # linear layer + TinyLoRA update
        R = (self.v @ self.P).view(self.rank, self.rank)
        return self.linear(x) + self.scaler * (x @ self.A @ R @ self.B)
