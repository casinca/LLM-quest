# Following RoFormer + YaRN paper for scaling, similar to @rasbt impl + HF's transformers impl
# This is a more explicit approach of RoPE vs Llama impl (which uses complex numbers directly)
import torch


class RoPE:
    @staticmethod
    def ntk_aware_base_scaling(theta_base, head_dim, ctx_len, old_ctx_len):
        """
        Fixed NTK aware scaling for theta base following @block97 and YaRN paper
        """
        return theta_base * (ctx_len / old_ctx_len) ** (head_dim / (head_dim - 2))

    @staticmethod
    def wavelength_scaling(base, head_dim, freq_cfg, ntk_aware_scaling=True):
        """
        Implements smooth frequency scaling with three bands (NTK by parts):
        - High frequency band: No interpolation (λ < ctx_len)
        - Medium frequency band: Smooth interpolation (λ ~ ctx_len)
        - Low frequency band: Full interpolation (λ > ctx_len)
            with λ = wavelength

        Args:
            base (int): Base value for frequency scaling
            head_dim (int): Dimension of each attention head, must be even
            freq_cfg (dict): Configuration for frequency scaling
            ntk_aware_scaling (bool, optional): Enable NTK aware scaling


        Returns:
            torch.Tensor: Modified theta values incorporating smooth frequency scaling across bands
        """
        # base frequencies
        if ntk_aware_scaling:
            base = RoPE.ntk_aware_base_scaling(base, head_dim, freq_cfg["ctx_len"], freq_cfg["og_ctx_len"])
        theta = 1 / base ** (2 * (torch.arange(0, head_dim // 2)) / head_dim)

        wavelen = 2 * torch.pi / theta

        # defining the ratio to complete a full cycle
        ratio = freq_cfg["og_ctx_len"] / wavelen

        # r > β, high freq = in window = no scaling
        # continue

        # low freq/long wavelen = long range = full interpolation by a scaled factor
        # r < α → γ=0 → h(θd) = θd/s
        scaled_theta = torch.where(
            ratio < freq_cfg["alpha"],
            theta / freq_cfg["factor"],
            theta,
        )

        # in between = medium freq = gradual/smooth transition interpolation
        # α <= r <= β → γ = (r − α) / (β − α)
        # clamp: Values outside [0, 1] would extrapolate rather than interpolate, defeating the purpose of smooth interp
        smooth_factor = ((ratio - freq_cfg["alpha"]) / (freq_cfg["beta"] - freq_cfg["alpha"])).clamp(0, 1)

        # h(θd) = (1 - γ) * (θd/s) + γ * θd
        smoothed_theta = (1 - smooth_factor) * (
            theta / freq_cfg["factor"]  # scaled component
        ) + smooth_factor * theta  # unscaled component

        # applying smooth interpolation to medium freq
        # α <= r <= β chained comparison don't work with tensors but only scalars, so using split comparison
        is_medium_freq = (ratio >= freq_cfg["alpha"]) & (ratio <= freq_cfg["beta"])
        final_theta = torch.where(is_medium_freq, smoothed_theta, scaled_theta)

        return final_theta

    @staticmethod
    def compute_angles(base, head_dim, ctx_len, smooth_scaling_cfg=None, ntk_aware_scaling=True):
        """
        Computes the sine and cosine of the angles for RoPE, optionally smooth frequency scaling.

        Args:
            base (int): The base value for the frequency calculation (default: 10000).
            head_dim (int): The dimension of each attention head, which must be an even number.
            ctx_len (int): The maximum context length.
            smooth_scaling_cfg (dict, optional): Configuration for YaRN smooth frequency scaling.
                If None, standard RoPE scaling is applied. Defaults to None.

        Returns:
            tuple (torch.Tensor, torch.Tensor): A tuple containing the cosine and sine of the computed angles.
                - cos: Tensor of shape (ctx_len, head_dim) representing the cosine of the angles.
                - sin: Tensor of shape (ctx_len, head_dim) representing the sine of the angles.
        """

        # even check for splitting head_dim in 2 for Θ = {θi = 10000^−2(i−1)/d, i ∈ [1, 2, ..., d/2]} with d=head_dim
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we have d/2 pairs of angles θi"

        # YaRN or classic RopE
        if smooth_scaling_cfg:
            theta = RoPE.wavelength_scaling(
                base,
                head_dim,
                smooth_scaling_cfg,
                ntk_aware_scaling,
            )
        else:
            theta = 1 / base ** (2 * (torch.arange(0, head_dim // 2)) / head_dim)

        positions = torch.arange(0, ctx_len)  # absolute position ("m" in the paper)

        # m ⊗ Θ, outer product to insert respective positions to all Θ in the matrix m0*vec0, m1*vec1, etc...
        angles = torch.outer(positions, theta)  # shape (ctx_len, head_dim //2)

        # expanding to head_dim (could have used angles.repeat(1, 2)) but cat is more explicit
        # unlike the paper we're not applying to contiguous pairs of embeddings but to embeddings split in half
        # thus we have to shape our angles the same way
        # ex: emb_dim=10 (Θ1...Θ5) (Θ1...Θ5) for (x1,...,x5) (x6,...,x10)
        # hence why we are concatenating and not interleaving
        angles = torch.cat([angles, angles], dim=-1)  # shape (ctx_len , head_dim)

        # final coordinate matrices → cos(mΘ) and sin(mΘ)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # we return 2 matrices cos and sin, needed for rotating our embeddings vectors
        return cos, sin

    @staticmethod
    def apply(x, cos, sin):
        """
        The goal here is to reshape x (input) to apply RoPE efficiently

        the paper applies to embeddings of contiguous pairs, ex if emb_dim=8 (x1,x2), (x3,x4), ...
        for efficiency, most implementations applies to halves (x1,x5), (x2,x6), ...

        Ex:
        → x_emb_dim = (x1,...,x4,x5,...,x8)

        emb_dim_h1 = (x1,...,x4)
        emb_dim_h2 = (x5,...,x8)
        → rotated = (-x5,...,-x8,x1,...,x4)

        x and rotated are now aligned with the cos and sin matrices
        So that we can easily apply our RoPE angles taking advantage of sparsity
        → rotated new coordinate x1' = cos(mΘ)*x1 + sin(mΘ)*(-x5)
                                x2'= cos(mΘ)*x2 + sin(mΘ)*(-x6)
                                ...
        → rotated new coordinate x5' = x1*sin(mΘ) + x5*cos(mΘ)
                                x6'= x2*sin(mΘ) + x6*cos(mΘ)
                                etc...

        with cos(mΘ) and sin(mΘ) from our computed_angles()
        """
        # x is queries/keys of shape (b, n_head, seq_len, head_dim)
        head_dim, seq_length = x.shape[-1], x.shape[2]
        # splitting embs in half (paper splits by pairs instead)
        h1 = x[..., : head_dim // 2]
        h2 = x[..., head_dim // 2 :]

        # preparing 2nd coordinates/embs matrix for calc optimization
        rotated = torch.concat((-h2, h1), dim=-1)
        # setting cos & sin up to seq_len. shape (ctx_len, head_dim) → (seq_len, head_dim)
        cos, sin = cos[:seq_length, :], sin[:seq_length, :]

        # apply RoPE efficiently
        res = cos * x + sin * rotated
        return res


# print(torch.arange(0, 20, 2))
# print(torch.arange(0, 20, 2)[: (20 // 2)])
# my_ver = 1 / 10000 ** (2 * (torch.arange(0, 8 // 2)) / 8)
# impl_ver = 1 / 10000 ** (torch.arange(0, 8, 2)[: 8 // 2] / 8)


# print(my_ver, "\n", impl_ver)
# print(my_ver.unsqueeze(0).shape, torch.arange(0, 8).unsqueeze(1).shape)

# Settings
# batch_size = 2
# context_len = 5
# num_heads = 4
# head_dim = 8
#
## Dummy query and key tensors
# torch.manual_seed(123)
# queries = torch.randn(batch_size, num_heads, context_len, head_dim)
# keys = torch.randn(batch_size, num_heads, context_len, head_dim)
#
## Instantiate RoPE parameters
# cos, sin = RoPE.compute_angles(base=10000, head_dim=head_dim, ctx_len=context_len)
# print(RoPE.apply(queries, cos, sin))
