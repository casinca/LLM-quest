# Following RoFormer + YaRN paper for scaling, similar to @rasbt impl + HF's transformers impl
# This is a more explicit approach of RoPE vs Llama impl (which uses complex numbers directly)
import torch


class RoPE:

    @staticmethod
    def partial_rotation(head_dim, factor):
        """
        # WARNING: potential source of divergence with HF when applied because:
        HF is rounding up(ceil) when calculating theta with torch.arange(0, head_dim, 2)
        While we are rounding down(floor) when calculating theta with torch.arange(0, head_dim // 2)

        So if the new partial head_dim is odd, I will at most rotate the specified factor while HF will rotate an extra
        dimension.
        Ex: head_dim = 6 rotation_factor = 0.5, head_dim = 3 → I will rotate 2 dimensions while HF will rotate 4.

        Helper function to scale the head dimension that will be used for rotating the dimensions/features

        Args:
            head_dim (int): The dimension of each attention head, which must be an even number.
            factor (float): The factor (0,1] to scale the head dimension by.
                            note: if new scaled head_dim is odd, will floor to closest even number

        returns:
            int: The scaled head dimension portion that will be used for rotating the dimensions/features
        """
        assert 0 < factor <= 1.0, "rotation factor must be greater than 0 and less than or equal to 1.0"
        return int(head_dim * factor)

    @staticmethod
    def ntk_aware_base_scaling(theta_base, head_dim, ctx_len, old_ctx_len):
        """
        Fixed NTK aware scaling for theta base, following @block97 and YaRN paper
        """
        return theta_base * (ctx_len / old_ctx_len) ** (head_dim / (head_dim - 2))

    @staticmethod
    def wavelength_scaling(base, head_dim, freq_cfg, ntk_aware_scaling=True, dtype=torch.float32):
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
            dtype (torch.dtype, optional): Data type for the angles computation


        Returns:
            torch.Tensor: Modified theta values incorporating smooth frequency scaling across bands
        """
        # base frequencies
        if ntk_aware_scaling:
            base = RoPE.ntk_aware_base_scaling(base, head_dim, freq_cfg["ctx_len"], freq_cfg["og_ctx_len"])
        theta = 1 / base ** (2 * (torch.arange(0, head_dim // 2, dtype=dtype)) / head_dim)

        wavelen = 2 * torch.pi / theta

        # defining the ratio to complete a full cycle
        ratio = freq_cfg["og_ctx_len"] / wavelen

        # r > β, high freq/short wavelen = in window = no scaling
        # continue

        # low freq/long wavelen = long range = full interpolation by a scaled factor
        # r < α → γ=0 → h(θd) = θd/s
        scaled_theta = torch.where(
            ratio < freq_cfg["alpha"],
            theta / freq_cfg["factor"],
            theta,
        )

        # in between = medium freq = gradual/smooth transition interpolation
        # α ≤ r ≤ β → γ = (r − α) / (β − α)
        # clamp: Values outside [0, 1] would extrapolate rather than interpolate, defeating the purpose of smooth interp
        smooth_factor = ((ratio - freq_cfg["alpha"]) / (freq_cfg["beta"] - freq_cfg["alpha"])).clamp(0, 1)

        # h(θd) = (1 - γ) * (θd/s) + γ * θd
        smoothed_theta = (1 - smooth_factor) * (
            theta / freq_cfg["factor"]  # scaled component
        ) + smooth_factor * theta  # unscaled component

        # applying smooth interpolation to medium freq
        # α ≤ r ≤ β chained comparison don't work with tensors but only scalars, so using split comparison
        is_medium_freq = (ratio >= freq_cfg["alpha"]) & (ratio <= freq_cfg["beta"])
        final_theta = torch.where(is_medium_freq, smoothed_theta, scaled_theta)

        return final_theta

    @staticmethod
    def compute_angles(
        base,
        head_dim,
        ctx_len,
        smooth_scaling_cfg=None,
        ntk_aware_scaling=True,
        rotation_factor=1.0,
        dtype=torch.float32,
    ):
        """
        WARNING: right now cos and sin are precomputed during model init, so no problem with operations in fp32 clashing
        with Pytorch autocast (AMP). But if we ever do something dynamic and cos and sin are calculated during forward
        pass, we we will have to explicitly disable autocast, ex:
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#L366C1-L378C60

        Computes the sine and cosine of the angles for RoPE, optionally smooth frequency scaling.

        Args:
            base (int): The base value for the frequency calculation.
            head_dim (int): The dimension of each attention head, which must be an even number.
            ctx_len (int): The maximum context length.
            smooth_scaling_cfg (dict, optional): Configuration for YaRN smooth frequency scaling.
                If None, standard RoPE scaling is applied. Defaults to None.
            ntk_aware_scaling (bool, optional): Enable NTK aware scaling. Defaults to True.
            rotation_factor (float, optional): The factor to scale the head dimension by, for partial rotation.
            dtype (torch.dtype, optional): Data type for the angles computation.

        Returns:
            tuple (torch.Tensor, torch.Tensor): A tuple containing the cosine and sine of the computed angles.
                - cos: Tensor of shape (ctx_len, head_dim) representing the cosine of the angles.
                - sin: Tensor of shape (ctx_len, head_dim) representing the sine of the angles.
        """

        # even check for splitting head_dim in 2 for Θ = {θi = 10000^−2(i−1)/d, i ∈ [1, 2, ..., d/2]} with d=head_dim
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we have d/2 pairs of angles θi"
        assert dtype == torch.float32, "for now enforcing dtype as float32 as arg rather than .float() again"

        if rotation_factor != 1.0:
            head_dim = RoPE.partial_rotation(head_dim, rotation_factor)

        # YaRN or classic RopE
        if smooth_scaling_cfg is not None:
            theta = RoPE.wavelength_scaling(
                base,
                head_dim,
                smooth_scaling_cfg,
                ntk_aware_scaling,
                dtype,
            )
        else:
            theta = 1.0 / base ** (2 * (torch.arange(0, head_dim // 2, dtype=dtype)) / head_dim)

        positions = torch.arange(0, ctx_len, dtype=dtype)  # absolute position ("m" in the paper)

        # m ⊗ Θ, outer product to insert respective positions to all Θ in the matrix m0*vec0, m1*vec1, etc...
        # (outer product of 2 vectors creates a matrix)
        angles = torch.outer(positions, theta)  # shape (ctx_len, head_dim //2)

        # expanding to head_dim (could have used angles.repeat(1, 2)) but cat is more explicit
        # unlike the paper we're not applying to contiguous pairs of dimensions but to dimensions split in half
        # thus we have to shape our angles the same way
        # ex: emb_dim=10 (Θ1...Θ5) (Θ1...Θ5) for (x1,...,x5) (x6,...,x10)
        # hence why we are concatenating and not interleaving
        angles = torch.cat([angles, angles], dim=-1)  # shape (ctx_len , head_dim)

        # final coordinate matrices → cos(mΘ) and sin(mΘ)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # we return 2 matrices cos and sin, needed for rotating our dimensions/features/head_dim vectors
        return cos, sin

    @staticmethod
    def _apply_partial_rope(x, cos, sin):
        """
        Applies partial RoPE to the input tensor x.
        Separate path in order to avoid repetitive useless splits and concatenations if unused/full RoPE.
        """
        seq_length = x.shape[2]
        rotation_dim = cos.shape[-1]
        # splitting x into rotated and unrotated parts
        x_rot, x_rest = x[..., :rotation_dim], x[..., rotation_dim:]

        # splitting dimensions/features in half (paper splits by pairs instead)
        h1 = x_rot[..., : rotation_dim // 2]
        h2 = x_rot[..., rotation_dim // 2 :]

        # preparing 2nd coordinates/dimensions matrix for calc optimization
        rotated = torch.concat((-h2, h1), dim=-1)
        # slicing cos & sin up to seq_len, shape (ctx_len, head_dim) → (seq_len, head_dim) and cast to x.dtype
        cos, sin = cos[:seq_length, :].to(x.dtype), sin[:seq_length, :].to(x.dtype)

        # apply RoPE efficiently (vectorized vs classic sparse paper)
        roped = cos * x_rot + sin * rotated

        # concat back rotated and unrotated dimensions/features into original head_dim
        res = torch.cat((roped, x_rest), dim=-1)

        return res

    @staticmethod
    def apply(x, cos, sin):
        """
        The goal here is to reshape x (input) to apply RoPE efficiently

        the paper applies to embeddings of contiguous pairs, ex if head_dim=8 (x1,x2), (x3,x4), ...
        for efficiency, most implementations applies to halves (x1,x5), (x2,x6), ...

        Ex:
        → x_head_dim          =        (x1,...,x4, x5,...,x8)
                                        ↑          ↑
        head_dim_h1 = (x1,...,x4)        |           |
        head_dim_h2 = (x5,...,x8)       ↓           ↓
        → rotated (-h2,h1)    =       (-x5,...,-x8, x1,...,x4)

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
        b, n_head, seq_length, head_dim = x.shape
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we need pairs"

        # If cos shape doesn't match x's head_dim, we infer that a partial RoPE should be applied
        if head_dim != cos.shape[-1]:
            return RoPE._apply_partial_rope(x, cos, sin)

        # Full RoPE
        else:
            # splitting dimensions/features in half (paper splits by pairs instead)
            h1 = x[..., : head_dim // 2]
            h2 = x[..., head_dim // 2 :]

            # preparing 2nd coordinates/dimensions matrix for calc optimization
            rotated = torch.concat((-h2, h1), dim=-1)
            # slicing cos & sin up to seq_len, shape (ctx_len, head_dim) → (seq_len, head_dim) and cast to x.dtype
            cos, sin = cos[:seq_length, :].to(x.dtype), sin[:seq_length, :].to(x.dtype)

            # apply RoPE efficiently (vectorized vs classic sparse paper)
            res = cos * x + sin * rotated

            return res


if __name__ == "__main__":

    batch_size = 1
    context_len = 5
    num_heads = 2
    head_dim = 6

    # Dummy query tensor
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)

    cos, sin = RoPE.compute_angles(
        base=10000,
        head_dim=head_dim,
        ctx_len=context_len,
        rotation_factor=0.7,  # if res is odd, will floor head_dim
        dtype=torch.float32,
    )
    print(queries)
    print(RoPE.apply(queries, cos, sin))
