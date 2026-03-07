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
        Ex: head_dim = 6 rotation_factor = 0.5, new_head_dim = 3 → I will rotate 2 dimensions while HF will rotate 4.

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
        https://github.com/huggingface/transformers/blob/02c324f43fe0ef5d484e846417e5f3bf4484524c/src/transformers/models/mixtral/modeling_mixtral.py#L212-L217

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
    def _apply_partial_rope(x, cos, sin, position_ids=None):
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
        if position_ids is not None:
            # cos/sin shape: (ctx_len, head_dim)
            # position_ids shape: (b, s) → gathered cos/sin: (b, s, head_dim) → unsqueezed: (b, 1, s, head_dim)
            cos = cos[position_ids].unsqueeze(1).to(x.dtype)
            sin = sin[position_ids].unsqueeze(1).to(x.dtype)
        else:
            # For non-KV cache case, slice from the beginning
            cos = cos[:seq_length, :].to(x.dtype)
            sin = sin[:seq_length, :].to(x.dtype)

        # apply RoPE efficiently (vectorized vs classic sparse paper)
        roped = cos * x_rot + sin * rotated

        # concat back rotated and unrotated dimensions/features into original head_dim
        res = torch.cat((roped, x_rest), dim=-1)

        return res

    @staticmethod
    def apply(x, cos, sin, position_ids=None):
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

        Args:
            x (torch.Tensor): The input tensor to apply RoPE to, shape (b, num_heads, seq_length, head_dim)
            cos (torch.Tensor): The cosine of the angles, shape (ctx_len, head_dim)
            sin (torch.Tensor): The sine of the angles, shape (ctx_len, head_dim)
            position_ids (torch.LongTensor, optional): Tensor of shape (batch_size, seq_len or 1 if KVcache) containing
                                                        the positions of each token. If None, applies RoPE to the first
                                                        seq_length positions (non-KV cache case).
        Returns:
            torch.Tensor: The input tensor with RoPE applied/rotated, shape (b, num_heads, seq_length, head_dim)
        """
        b, n_head, seq_length, head_dim = x.shape
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we need pairs"

        # If precomputed cos shape doesn't match x's head_dim, we infer that a partial RoPE should be returned
        if head_dim != cos.shape[-1]:
            return RoPE._apply_partial_rope(x, cos, sin, position_ids)

        # Full RoPE
        # splitting dimensions/features in half (paper splits by pairs instead)
        h1 = x[..., : head_dim // 2]
        h2 = x[..., head_dim // 2 :]

        # preparing 2nd coordinates/dimensions matrix for calc optimization
        rotated = torch.concat((-h2, h1), dim=-1)

        if position_ids is not None:
            # cos/sin shape: (ctx_len, head_dim)
            # position_ids shape: (b, s) → gathered cos/sin: (b, s, head_dim) → unsqueezed: (b, 1, s, head_dim)
            cos = cos[position_ids].unsqueeze(1).to(x.dtype)
            sin = sin[position_ids].unsqueeze(1).to(x.dtype)
        else:
            # For non-KV cache case, slice from the beginning
            cos = cos[:seq_length, :].to(x.dtype)
            sin = sin[:seq_length, :].to(x.dtype)

        # apply RoPE efficiently (vectorized vs classic sparse paper)
        res = cos * x + sin * rotated

        return res


class VisionRoPE:
    """
    This 2D variant is also known as Axial 2D RoPE, the RoPE-Mixed paper (https://arxiv.org/abs/2403.13298) mentions the
    EVA-02 paper (Evangelion nod) as a reference for the Axial 2D RoPE: https://arxiv.org/abs/2303.11331

    NOTE: For simplicity, for fixed-size images so that we can just precompute the same way as we do for text in the
    `RoPE` class.

    Same as 1D text classic RoPE except we are also adding a new direction/y-axis for images. We can see, in this
    context, the classic text RoPE as already a 1D x-axis (as a sequence)

    Why do we need 2D RopE, with a good example:

    Let's take an image divided in 3x3 grid of patches:
    [0] [1] [2]
    [3] [4] [5]
    [6] [7] [8]

    When it's flattened (after `PatchEmbedding3D`) we have a 1D sequence of patches for the transformer block:
    [0, 1, 2, 3, 4, 5, 6, 7, 8]

    If we use classic text RoPE, we would only be encoding the position along this flat sequence:

    - patch [2] and patch [3] are next to each other in the flat sequence but in reality, in the image, they are 3
        step away from each other (if we take manhattan distance).
    - Inversely and similarly, patch [0] and [3] are next to each other (vertically) in the image but not in the flat
        sequence.

    Hence the reason to use 2D RoPE and preserve this true spacial distance information.

    We will split the `head_dim` into 2 parts, one for row (x-axis) and one for column (y-axis).
    Instead of having position ids for a single sequence, we will have a grid of positions.

    For example, patch [0] would be (0,0) and patch [3] (1,0):
        - Their row x-axis RoPE (same as 1D classic text RoPE) are both 0 and 0 (could be seen as a start of a sequence)
        - Their column y-axis RoPE are 0 and 1, effectively recording that they are 1 step apart on the y-axis
    """

    @staticmethod
    def compute_angles_2d(
        base,
        head_dim,
        height_patches,
        width_patches,
        num_frames=1,
        dtype=torch.float32,
    ):
        """
        Computes 2D RoPE angles

        We omit YaRN scaling

        Args:
            base (int): Base value for frequency scaling.
            head_dim (int): Dimension of each attention head (must be divisible by 4).
            height_patches (int): Fixed number of patches along the height (H).
            width_patches (int): Fixed number of patches along the width (W).
            num_frames (int): Number of temporal frames (default 1 for images, videos would be >1).
            dtype (torch.dtype): Data type for computation.

        Returns:
            tuple: (cos, sin) tensors of shape (num_frames * height_patches * width_patches, head_dim)
        """
        # Half of head_dim is going to y (row) and half to x (col)
        # since we also need pairs for cos/sin, head_dim must be divisible by 4.
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        # Same as text RoPE: compute 1D theta/frequencies, except now the range is half a head dim
        # we're doing this once (not each axis) because frequencies are the same for both axes.
        half_dim = head_dim // 2
        theta = 1.0 / (base ** (2 * torch.arange(0, half_dim // 2, dtype=dtype) / half_dim))

        # Generate the 2D coordinates/distances using PyTorch meshgrid
        # this gives us the (row, col) for every patch in the H x W image grid.
        #
        # Example to remember what it does:
        #
        # with an image split in a 2x2 grid of 4 patches (ie `height_patches=2` and `width_patches=2`)
        # feeding 2x torch.arange for x and y-axis, will give 2x 2D tensors:
        #
        # row_pos = [[0, 0],   # Row 0
        #           [1, 1]]   # Row 1
        #
        # col_pos = [[0, 1],   # Col 0, Col 1
        #           [0, 1]]   # Col 0, Col 1
        row_pos, col_pos = torch.meshgrid(
            torch.arange(height_patches, dtype=dtype),
            torch.arange(width_patches, dtype=dtype),
            indexing="ij",  # row-major order (left-to-right, top-to-bottom), just like conv3D flattening
        )

        # Flatten the 2x 2D grid into the 1D sequence the transformer block expects
        # shape each: (height_patches, width_patches) → (height_patches * width_patches)
        flat_row_pos = row_pos.flatten()
        flat_col_pos = col_pos.flatten()

        # Same as Text RoPE but twice (for each axis): multiply positions by theta to get the angles (outer product)
        # shape of both: (height_patches * width_patches, half_dim // 2)
        angles_y = torch.outer(flat_row_pos, theta)
        angles_x = torch.outer(flat_col_pos, theta)

        # concatenate the y and x angles together, shape: (height_patches * width_patches, half_dim)
        angles_2d = torch.cat([angles_y, angles_x], dim=-1)

        # If it's a video, repeat the spatial layout `num_frames` times
        # shape: (num_frames * height_patches * width_patches, half_dim)
        # NOTE: yes, we are duplicating
        # There's no 3D/time/temporal awareness "here", ie if the patches are from image1 or image2
        # It's purely 2D/spatial awareness per image/independently.
        # Temporal awareness would be handled by MRoPE (Multimodal RoPE) in the Multimodal model
        if num_frames > 1:
            angles_2d = angles_2d.repeat(num_frames, 1)

        # Same as Text RoPE now:
        # duplicate to match the full head_dim for apply()
        # shape: (num_frames * height_patches * width_patches, head_dim)
        angles = torch.cat([angles_2d, angles_2d], dim=-1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    @staticmethod
    def apply(x, cos, sin, position_ids=None):
        """
        NOTE: This is the same thing/copy as `RoPE.apply()` method. Nothing change as we use fixed size image.

        Args:
            x (torch.Tensor): The input tensor to apply RoPE to, shape (b, num_heads, seq_length, head_dim)
            cos (torch.Tensor): The cosine of the angles, shape (ctx_len, head_dim)
            sin (torch.Tensor): The sine of the angles, shape (ctx_len, head_dim)
            position_ids (torch.LongTensor, optional): Tensor of shape (batch_size, seq_len or 1 if KVcache) containing
                                                        the positions of each token. If None, applies RoPE to the first
                                                        seq_length positions (non-KV cache case).
        Returns:
            torch.Tensor: The input tensor with RoPE applied/rotated, shape (b, num_heads, seq_length, head_dim)
        """
        b, n_head, seq_length, head_dim = x.shape
        assert head_dim % 2 == 0, "head dim must be divisible by 2 as we need pairs"

        # Full RoPE
        # splitting dimensions/features in half (paper splits by pairs instead)
        h1 = x[..., : head_dim // 2]
        h2 = x[..., head_dim // 2 :]

        # preparing 2nd coordinates/dimensions matrix for calc optimization
        rotated = torch.concat((-h2, h1), dim=-1)

        if position_ids is not None:
            # cos/sin shape: (ctx_len, head_dim)
            # position_ids shape: (b, s) → gathered cos/sin: (b, s, head_dim) → unsqueezed: (b, 1, s, head_dim)
            cos = cos[position_ids].unsqueeze(1).to(x.dtype)
            sin = sin[position_ids].unsqueeze(1).to(x.dtype)
        else:
            # For non-KV cache case, slice from the beginning
            cos = cos[:seq_length, :].to(x.dtype)
            sin = sin[:seq_length, :].to(x.dtype)

        # apply RoPE efficiently (vectorized vs classic sparse paper)
        res = cos * x + sin * rotated

        return res


if __name__ == "__main__":
    torch.manual_seed(123)

    batch_size = 2
    num_heads = 2
    head_dim = 6
    # Dummy test for 1D classic text RoPE
    context_len = 5
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)

    cos, sin = RoPE.compute_angles(
        base=10000,
        head_dim=head_dim,
        ctx_len=context_len,
        rotation_factor=0.7,  # if partial rotation head_dim is odd, will floor to head_dim
        dtype=torch.float32,
    )
    print("RoPE input:", queries)
    rotated_queries = RoPE.apply(queries, cos, sin)
    print("RoPE output:", rotated_queries)
    print("RoPE output shape:", rotated_queries.shape)

    # Dummy test for VisionRoPE (Axial 2D RoPE)
    # a single image/frame of a 2x3 grid of patches (H=2, W=3)
    head_dim = 4
    num_frames = 1
    height_patches = 2
    width_patches = 3
    flat_3d_seq_len = num_frames * height_patches * width_patches

    vision_queries = torch.randn(batch_size, num_heads, flat_3d_seq_len, head_dim)

    cos_vision, sin_vision = VisionRoPE.compute_angles_2d(
        base=10000,
        head_dim=head_dim,
        height_patches=height_patches,
        width_patches=width_patches,
        num_frames=num_frames,
        dtype=torch.float32,
    )
    print("VisionRoPE input:", vision_queries)
    rotated_vision_queries = VisionRoPE.apply(vision_queries, cos_vision, sin_vision)
    print("VisionRoPE output:", rotated_vision_queries)
    print("VisionRoPE output shape:", rotated_vision_queries.shape)
