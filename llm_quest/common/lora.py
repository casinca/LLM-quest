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
        d (int): Input dimension (rows of W)
        k (int): Output dimension (columns of W)
        r (int): Rank of the low-rank matrices (r << d, k)
        alpha (float): Scaling factor for the update
        linear_bias (bool): Whether to add a bias to the linear layer

    Returns:
        torch.Tensor: The output tensor after the forward pass: xW^T + α/r * xAB.
    """

    def __init__(self, d, k, r, alpha, linear_bias=False):
        super().__init__()
        self.linear = nn.Linear(d, k, bias=linear_bias)
        # zero init for B (nn.param flags the tensor as a learnable param)
        self.B = nn.Parameter(torch.zeros(r, k))

        # random Gaussian init for A (per the paper)
        # LoRA code from MSFT and @rasbt's alt: nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.A = nn.Parameter(torch.empty(d, r))
        nn.init.normal_(self.A, mean=0.0, std=0.02)  # small std commonly used

        # AB is scaled by α/r, where α is a constant scaling factor: α/r * xAB
        # (α is basically the same as the 'α' factor in Adam to tune the lr)
        self.scaler = alpha / r

    def forward(self, x):
        # return standard linear layer + our low-rank parametrized update matrices α/r * xAB (LoRA)
        return self.linear(x) + self.scaler * (x @ self.A @ self.B)


# for completeness (will only change Attention Q,K,V and output layer for instruct training like the paper)
# (iterating through model.modules() is faster and doesn't need explicit dfs but more verbose than named.children())
def replace_with_lora(model, rank, alpha):
    """
    This function iterates through all modules in the given model. When it finds a nn.Linear layer,
    it replaces it with a LoRALinearLayer that has the same input and output dimensions,
    with the specified rank and alpha for the LoRA update.
    For modules that are not nn.Linear, it recursively calls itself to check and replace
    any nn.Linear layers within those submodules.

    Args:
        model (nn.Module): The model in which to replace layers.
        rank (int): The rank 'r' for the low-rank update in LoRA.
        alpha (float): The scaling factor for the low-rank update.
    """
    # iterating through modules, ex: ("linear1", nn.Linear(10, 5)), ("relu", nn.ReLU())
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # replace Linear with LoRA:
            d_in = module.in_features
            d_out = module.out_features
            bias = module.bias is not None
            setattr(model, name, LoRALinearLayer(d_in, d_out, rank, alpha, linear_bias=bias))
            # setattr() same as model.name = LoRALinearLayer(d_in, d_out, rank, alpha, linear_bias=bias)
        else:
            # recursively check & replace in submodules
            replace_with_lora(module, rank, alpha)


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
        # in case
        for param in self.linear.parameters():
            param.requires_grad = False

        # Compute truncated SVD of the trained weight/matrix
        # W = U @ Σ @ V^T, with W as shape (d, k) so we transpose because nn.Linear stores weight as (k, d)
        with torch.no_grad():
            # can't use "Σ" symbol, so "S" for sigmas. Also doing SVD in fp32
            U, S, Vt = torch.linalg.svd(self.linear.weight.data.float().T, full_matrices=False)

            # truncating: top r singular vectors and values
            U_r = U[:, :r]  # (d, r)
            S_r = S[:r]  # (r,) PyTorch for efficiency returns directly a vector, instead of a diag matrix with 0s
            Vt_r = Vt[:r, :]  # (r, k)

            # Unlike LoRA, AB are frozen, not trained (registered as buffers), only R is trained
            self.register_buffer("A", U_r @ torch.diag(S_r).to(self.linear.weight))  # A = U_r * Σ_r (shape: d, r)
            self.register_buffer("B", Vt_r.to(self.linear.weight))  # B = V_r^T (shape: r, k)

        # Normal init N(0, σ²) with σ = 1e-5 per the paper
        self.R = nn.Parameter(torch.empty(r, r))
        nn.init.normal_(self.R, mean=0.0, std=1e-5)

        # paper mentions p.12: α = r for instruction tuning (so scaler = 1), α = 16 for GLUE benchmark
        self.scaler = alpha / r

    def forward(self, x):
        """Forward pass: h = xW^T + α/r * xARB"""
        # linear layer + LoRA-XS update
        return self.linear(x) + self.scaler * (x @ self.A @ self.R @ self.B)
