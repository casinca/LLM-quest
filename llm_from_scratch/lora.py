import torch
import torch.nn as nn


# TODO impl weight merging for inference optimization?
class LoRALinearLayer(nn.Module):
    """
    LoRA changes the default linear layers (paper mentions Attention weights, but any learnable weights) to add 2
    lower rank matrices BA for learning (instead of directly updating W of shape (d_in, d_out) with a ∆W of same size).
    The update isn't done to W (frozen) but separately learned through BA.

    The intuition is that we can replace a full weight update matrix ∆W by approximating it with 2 low-rank matrices.
    That's because fine-tuning updates are generally close to low-rank, hence we can explicitly learn these updates to
    the original model in their factorized form.

    ie, our new forward pass becomes Wx → Wx + BAx where W is frozen and gradient flows through BA instead of ∆W.
    with W of shape (d, k) and B, A of respective shapes (d, r) and (r, k)
    With r a rank based on num of params and weights (r ∈ {1, 2, 4, ..., 2^n} a good starting point per the paper)

    The fact that gradient updates with BA are decoupled from the weights, gives us the possibility to store and
    reuse any BA for another downstream task at no additional inference latency (vs re-finetuning).
    ex: Wx - BA to get back default pretrained weights and Wx + B'A' to use another set of updated weights.

    Args:
        d (int): Input dimension (rows of W)
        k (int): Output dimension (columns of W)
        r (int): Rank of the low-rank matrices (r << d, k)
        alpha (float): Scaling factor for the update

    Returns:
        torch.Tensor: The output tensor after applying the LoRA update, which is the result of α/r * BAx.
    """

    def __init__(self, d, k, r, alpha, linear_bias=False) -> None:
        super().__init__()
        self.linear = nn.Linear(d, k, bias=linear_bias is not None)
        # zero init for B (nn.param flags the tensor as a learnable param)
        self.B = nn.Parameter(torch.zeros(d, r))

        # random Gaussian init for A (per the paper)
        # LoRA code from MSFT and @rasbt's alt: nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.A = nn.Parameter(torch.empty(r, k))
        nn.init.normal_(self.A, mean=0.0, std=0.02)  # small std commonly used

        # BA is scaled by α/r, where α is a constant scaling factor: α/r * BAx
        # (α is basically the same as the 'α' factor in Adam to tune the lr)
        self.scaler = alpha / r

    def forward(self, x):
        # return standard linear layer + our low-rank parametrized update matrices α/r * xBA (LoRA)
        return self.linear(x) + self.scaler * (x @ self.B @ self.A)


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
