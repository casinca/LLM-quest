import torch
import torch.nn as nn

from llm_quest.common.rope import VisionRoPE

# NOTE: interchanging "frame" and "image" in comments and possibly variables, same thing.

# Some differences with our `PatchEmbedding2D` class for the ViT in multimodal/vision_transformer/vit_model.py:
#
# For the from-scratch ViT, we were only using fixed size square images (CxHxW) where Height and Width are the same and
# C is the number of channels (3 for RGB). hence `2D` for HxW.
# And we had for entry 4D tensors (batch, channels, height, width).
#
# Qwen3.5 VL is more SOTA, and also includes videos, not just single images. (We'll still assume fixed square size for
# simplicity).
# Videos are just a sequence of images, the sequence length is the number of images/frames in the video.
#
# Therefore, we need to add a new dimension: the temporal/time dimension (T), which represents the number of images in
# the video. We could just see it as a minibatch dimension of images (if we see a video as a batch of images).
#
# So now our entry tensors are 5D, to account for the added temporal dimension of videos:
# (batch, channels, time, height, width).
# If we have a single batch of an RGB video (of 8 32x32 images), then our 5D tensor would be (1,3,8,32,32).
# If we have a single RGB image (can be seen as a video of 1 image/frame), then T=1, (1,3,1,32,32).
#
#
# On top of that, instead of having independent patches per image, we can also merge patches of a video together, into a
# single embedding.
#
# Eg, if we have a video of 8 frames/images we can group images 1+2, 3+4, 5+6, and 7+8. The time dimension becomes 4
# from 8. Instead of outputting 8 separate grids of patches, it outputs 4 grids of patches.
# This is controlled by the `temporal_patch_size` argument.
# In the example above, `temporal_patch_size=2`, 8/2=4.
# If we wanted each frame processed entirely independently, then `temporal_patch_size=1`.
# We can also keep `temporal_patch_size=2` with a single image and duplicate it. Which simulate a video of the same
# image/frame.
#
# Overall this merging of images give a sense of motion/spatio-temporal information but is also very useful for
# efficiency since we have less embeddings/number of patches to process.
#
# These are the main differences for `PatchEmbedding3D` class vs `PatchEmbedding2D` besides that the convolution is now
# in 3D (time, height, width) instead of 2D (height, width).
#
# Also since we are not doing simple classification tasks, there's no more CLS token prepended.


class PatchEmbedding3D(nn.Module):
    """
    Convert videos (sequence of images) or image tensors to sequence of patch embeddings.
    Similar to `PatchEmbedding2D` class but we use a 3D convolution to handle the temporal dimension.

    This class:
    1. Splits image into patches
    2. Linear projection of flattened patches to embedding dimension
    (nothing is prepended like for the classification ViT)

    Args:
        img_height (int): Input image height
        img_width (int): Input image width
        num_channels (int): Number of input channels (RGB = 3)
        emb_dim (int): Embedding dimension
        patch_size (int): Size of each patch (assumes square patches)
        temporal_patch_size (int): number of images/frames to group together, if >1, will reduce the time dimension
    """

    def __init__(self, img_width, img_height, num_channels, emb_dim, patch_size, temporal_patch_size):
        super().__init__()
        assert img_width % patch_size == 0, f"Image width {img_width} not divisible by patch size {patch_size}"
        assert img_height % patch_size == 0, f"Image height {img_height} not divisible by patch size {patch_size}"

        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_per_image = (img_width * img_height) // patch_size**2  # N = HW/P^2
        # total_num_patches is now = self.num_patches_per_image * time // temporal_patch_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)  # ie (time, patch_h, patch_w)
        self.conv_proj = nn.Conv3d(
            in_channels=num_channels,
            out_channels=emb_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,  # default value but being explicit
            bias=True,  # default value but being explicit
        )

    def forward(self, x):
        """
        Args:
            x: input of shape (b, n_channels, time, img_h, img_w)

        Returns:
            patch embeddings, shape (b, n_patches, emb_dim)
        """
        b, n_channels, time, img_h, img_w = x.shape
        assert img_h == self.img_height and img_w == self.img_width, (
            f"Input image shape {x.shape} does not match expected shape {self.img_height}x{self.img_width}"
        )
        assert time % self.temporal_patch_size == 0, (
            f"Input time shape {time} is not divisible by temporal_patch_size {self.temporal_patch_size}"
        )
        # Fixed resolution for simplicity
        # shape: (b, n_channels, time, img_h, img_w) → (b, emb_dim, time, num_patches_h, num_patches_w)
        x = self.conv_proj(x)
        # flattening & transposing → (b, n_patches, emb_dim)
        x = x.flatten(2).transpose(1, 2)

        return x


class Qwen3_5VisionFFN(nn.Module):
    """
    Same as our ViT `FFN` class but:
    - GELU is approximated with tanh (using Pytorch) instead of exact GELU.
    """

    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg["vision_emb_dim"], cfg["vision_hidden_dim"])
        self.lin2 = nn.Linear(cfg["vision_hidden_dim"], cfg["vision_emb_dim"])
        self.activ = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.lin2(self.activ(self.lin1(x)))


class Qwen3_5VisionAttention(nn.Module):
    """
    Bidirectional multi-head attention for the Qwen vision model (which is based on Qwen3-VL) with 2D axial RoPE.

    This is adapted from our `ViTMultiHeadAttention` (`llm_quest/multimodal/vision_transformer/vit_attention.py`)
    with these differences:
    - Adds Axial 2D RoPE for queries and keys
    - QKV have biases
    - No dropout (not training, loading weights)

    Args:
        cfg(dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__()
        # Qwen chose d_in = d_out, also called hidden_size in HF
        self.d_in = cfg["vision_emb_dim"]
        self.num_heads = cfg["vision_num_heads"]
        self.head_dim = self.d_in // self.num_heads

        # Fused QKV with bias
        self.qkv = nn.Linear(self.d_in, self.d_in * 3, bias=True)
        self.proj = nn.Linear(self.d_in, self.d_in, bias=True)

    def forward(self, x, cos, sin):
        """
        Args:
            x: (batch, seq_len, d_in)
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)

        `seq_len` is the same as `num_patches` in this context

        Returns:
            (batch, seq_len, d_out)
        """
        b, seq_len, d_in = x.shape

        # split back and reshape to b,h,s,d
        qkv = self.qkv(x)
        queries, keys, values = qkv.chunk(3, dim=-1)

        queries, keys, values = map(
            lambda t: t.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2), (queries, keys, values)
        )

        # apply vision/axial 2D RoPE
        queries = VisionRoPE.apply(queries, cos, sin)
        keys = VisionRoPE.apply(keys, cos, sin)

        # Bidirectional attention (no causal mask, same as our classification ViT)
        ctx_tensor = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            is_causal=False,
        )

        # shape (batch, num_heads, seq_len, head_dim) → (batch, seq_len, d_in)
        ctx_tensor = ctx_tensor.transpose(1, 2).contiguous().view(b, seq_len, self.d_in)
        ctx_tensor = self.proj(ctx_tensor)

        return ctx_tensor


class Qwen3_5VisionTransformerBlock(nn.Module):
    """
    Qwen3.5 Vision Transformer Block.

    Same structure as our classification ViT's `ViTTransformerBlock`:
        LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

    Differences from our classification ViT trf block:
    - Uses `Qwen3_5VisionAttention` (with RoPE) instead of `ViTMultiHeadAttention`
    - Uses `Qwen3_5VisionFFN` (GELU approx w/ tanh) instead of FFN
    - Uses `nn.LayerNorm` directly for simplicity

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["vision_emb_dim"], eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg["vision_emb_dim"], eps=1e-6)
        self.att = Qwen3_5VisionAttention(cfg)
        self.ffn = Qwen3_5VisionFFN(cfg)

    def forward(self, x, cos, sin):
        """
        Args:
            x: (batch, seq_len, d_in)
            cos: (seq_len, head_dim) angles for attention (2D axial RoPE)
            sin: (seq_len, head_dim) angles for attention (2D axial RoPE)

        Returns:
            (batch, seq_len, d_in)
        """
        residual = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class Qwen3_5VisionModel(nn.Module):
    """
    Complete Vision model for the final Multimodal Qwen3.5.
    This is the equivalent of our `ViTModel` for the multimodal GPT-2 pipeline.

    NOTE: Even if we are doing fixed size images/frames, the number of frames (the time dimension) is completely
    variable and dynamic, the ViT has no temporal awareness (except in conv3D if temporal_patch_size > 1):
    We can pass a single 768x768 image, just like a video of 10x 768x768 images.

    - `PatchEmbed3D`: image → patch embeddings
    - add Positional embeddings (yes kept from the original ViT, on top of 2D RoPE)
    - use 2D spatial RoPE for attention
    - transformer blocks
    - `ViTMergeAdapter`: spatial merge (optional) + project to text/llm emb dim

    Args:
        cfg (dict): Vision configuration dictionary with keys:
            emb_dim(int): embedding dimension = hidden_size = d_in = d_out for Qwen3 Vision (not emb dim of the text/LLM)
            img_width(int): input image width
            img_height(int): input image height
            patch_size(int): size of each patch (assumes square patches)
            in_channels(int): number of input channels (RGB = 3)
            num_position_embeddings(int): max number of patches in the image/frame (capacity of the position embeddings)
            num_heads(int): number of attention heads
            temporal_patch_size(int): number of images/frames to merge together, if >1, will reduce the time dimension
            spatial_merge_size(int): number of patches to merge per side (height and width) within images
            llm_d_in(int): text model/LLM embedding dimension (output of the ViT (after adapter) / input dim of the LLM)
    """

    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg["vision_emb_dim"]
        n_layers = cfg["vision_n_layers"]
        num_heads = cfg["vision_num_heads"]
        rope_base = cfg["vision_rope_base"]
        llm_d_in = cfg["llm_d_in"]  # also called out_hidden_size = llm/text model emb_dim

        img_width = cfg["img_width"]
        img_height = cfg["img_height"]
        patch_size = cfg["patch_size"]

        assert img_width % patch_size == 0, f"Image width {img_width} not divisible by patch size {patch_size}"
        assert img_height % patch_size == 0, f"Image height {img_height} not divisible by patch size {patch_size}"
        self.n_width_patches = img_width // patch_size
        self.n_height_patches = img_height // patch_size
        self.n_spatial_patches = self.n_width_patches * self.n_height_patches

        # Since we are doing fixed size, we expect the number of patches per image to be less than the maximum capacity
        # of the model. With variable size, the max capacity is used to downsize larger images.
        assert self.n_spatial_patches <= cfg["num_position_embeddings"], (
            f"the image size {img_width}x{img_height} "
            f"is too large for the number of position embeddings {cfg['num_position_embeddings']}"
        )

        self.patch_embed = PatchEmbedding3D(
            img_width=img_width,
            img_height=img_height,
            num_channels=cfg["in_channels"],
            emb_dim=emb_dim,
            patch_size=patch_size,
            temporal_patch_size=cfg["temporal_patch_size"],
        )

        # Learned positional embeddings (spatial only, repeated across frames) will be overwritten by loaded weights
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_spatial_patches, emb_dim))

        # Vision-specific RoPE (separate from text model's RoPE)
        # We precompute angles for Vision RoPE
        # NOTE since precomputed, we expect fixed size images/patches, not handling variable length sequences here.
        # Concerning `num_frames=1`, instead of storing duplicated angles for each fixed size frame, we repeat in the
        # forward pass for memory efficiency.
        cos, sin = VisionRoPE.compute_angles_2d(
            base=rope_base,
            head_dim=emb_dim // num_heads,
            height_patches=self.n_height_patches,
            width_patches=self.n_width_patches,
            num_frames=1,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList([Qwen3_5VisionTransformerBlock(cfg=cfg) for _ in range(n_layers)])

        # Head is: spatial merge + project to text model dim
        self.merge_adapter = ViTMergeAdapter(
            vit_d_out=emb_dim,
            llm_d_in=llm_d_in,
            n_height_patches=self.n_height_patches,
            n_width_patches=self.n_width_patches,
            spatial_merge_size=cfg["spatial_merge_size"],
        )

    def forward(self, x):
        """
        Process images through the vision encoder.

        Args:
            x: (batch, in_channels, time, img_h, img_w) Pre-processed image pixels for the `PatchEmbedding3D`

        Returns:
            (batch, num_merged_patches, llm_d_in) vision embeddings in text embedding space
        """
        # extract patches via 3D convolution, shape (batch, num_patches, emb_dim)
        x = self.patch_embed(x)

        seq_len = x.shape[1]  # flat total number of patches
        # "actual" number of frames/images, because if temporal_patch_size > 1, actual_n_frames < time/initial n_frames.
        actual_n_frames = seq_len // self.n_spatial_patches

        # add positional embeddings (repeated across time frames)
        pos_embed_repeated = self.pos_embed.repeat(1, actual_n_frames, 1)
        x = x + pos_embed_repeated[:, :seq_len, :]

        # repeat cos and sin for each time frame
        batch_cos = self.cos.repeat(actual_n_frames, 1)  # shape (actual_n_frames*n_spatial_patches, head_dim)
        batch_sin = self.sin.repeat(actual_n_frames, 1)

        for block in self.blocks:
            x = block(x, batch_cos, batch_sin)  # shape (batch, total_num_patches/seq_len, emb_dim)

        # Merge patches within images (reduces by spatial_merge_size² = 2x2) and project to text dim
        merged_output = self.merge_adapter(x)

        return merged_output


class ViTMergeAdapter(nn.Module):
    """
    The `ViTMergeAdapter` is more complex than the `ViTAdapter` because it also performs spatial downsampling, reducing
    the number of patches per frame/image, whereas the `ViTAdapter` only does feature reprojection.

    Merges adjacent patches if `spatial_merge_size` > 1 and reprojects to text embedding dimension.

    There are similarities concerning this compression and the `PatchEmbedding3D` class.
    In the `PatchEmbedding3D` class, we have the `temporal_patch_size` argument to control the compression over the
    time dimension.

    Here, the compression, controlled by the `spatial_merge_size` argument, is over the spatial dimension. We are not
    reducing the number of images in a video, but rather patches within image, in both directions (height and width).

    Takes groups of "spatial_merge_size²" (Qwen config uses 2x2 = 4) adjacent patches, concatenates their features, and
    projects through an FFN to the text model's embedding dimension:

    vision_emb_dim → (merge 4 patches) → vision_emb_dim * 4 → FFN → llm_d_in/text_emb_dim

    Args:
        vit_d_out (int): Vision hidden dimension (per patch)
        llm_d_in (int): Text model embedding dimension (output)
        spatial_merge_size (int): Number of patches to merge per side (2 → 2x2=4 patches merged)
    """

    def __init__(self, vit_d_out, llm_d_in, n_height_patches, n_width_patches, spatial_merge_size=2):
        super().__init__()
        self.m = spatial_merge_size
        self.n_h_patches = n_height_patches
        self.n_w_patches = n_width_patches
        self.merged_size = vit_d_out * (self.m**2)  # 4 patches concatenated

        # (not using nn.Sequential module, it'll be easier for loading weights)
        self.norm = nn.LayerNorm(vit_d_out, eps=1e-6)
        self.lin1 = nn.Linear(self.merged_size, self.merged_size)
        self.activ = nn.GELU()
        self.lin2 = nn.Linear(self.merged_size, llm_d_in)

    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, vit_d_out) arranged in row major order (from conv3d in PatchEmbedding3D)

        Returns:
            (batch, num_merged_patches, llm_d_in) projected to text embedding dim
        """
        b, n_patches, vit_d_out = x.shape
        t = n_patches // (self.n_h_patches * self.n_w_patches)  # time dim (actual num of frames)

        x = self.norm(x)  # norm, before merging

        # Group 2x2 patches: reshape to grid, reorder so each 2x2 block is contiguous, then flatten
        x = x.view(b, t, self.n_h_patches // self.m, self.m, self.n_w_patches // self.m, self.m, vit_d_out)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # shape (b, t, block_h, block_w, m, m, vit_d_out)
        x = x.view(b, -1, self.merged_size)  # shape (b, num_merged_patches, merged_size)

        x = self.lin2(self.activ(self.lin1(x)))  # shape (b, num_merged_patches, llm_d_in)

        return x


# quick inline test
if __name__ == "__main__":
    torch.manual_seed(123)

    x = torch.randn(1, 3, 8, 32, 32)
    patch_size = 8
    img_h, img_w = 32, 32
    temporal_patch_size = 2
    num_frames = 8 // temporal_patch_size
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    emb_dim = 128

    patch_embedding = PatchEmbedding3D(
        img_width=img_w,
        img_height=img_h,
        num_channels=3,
        emb_dim=emb_dim,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
    )
    x_shaped = patch_embedding(x)
    print(x_shaped.shape)  # (1, 64, 128)
    # time is 8/temporal_patch_size = 4
    # num_patches h and w (since square) is 32/patch_size = 4 for both,
    # emb_dim is 128
    # so (1, 4*4*4, 128) = (1, 64, 128)

    #  --- dummy test for Qwen3_5VisionAttention ---
    dummy_cfg = {
        "vision_emb_dim": emb_dim,  # d_in = d_out = hidden_size
        "vision_num_heads": 4,
    }
    cos, sin = VisionRoPE.compute_angles_2d(
        base=10000,
        head_dim=dummy_cfg["vision_emb_dim"] // dummy_cfg["vision_num_heads"],
        height_patches=num_patches_h,
        width_patches=num_patches_w,
        num_frames=num_frames,
    )

    attn = Qwen3_5VisionAttention(dummy_cfg)
    y = attn(x_shaped, cos, sin)
    print("Qwen3_5VisionAttention output shape:", y.shape)
    print(y)

    # Test with a simple vision config (matching ~Qwen3.5-0.8B vision)
    test_cfg = {
        "vision_n_layers": 2,
        "vision_emb_dim": 768,
        "vision_hidden_dim": 3072,
        "vision_num_heads": 12,
        "vision_rope_base": 10000,
        "spatial_merge_size": 2,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "img_width": 384,
        "img_height": 384,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "llm_d_in": 1024,
    }

    model = Qwen3_5VisionModel(test_cfg)
    print(f"Vision model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate a single 384x384 image video of length 2 (1 batch, 3 channels, 2 time, 384 H, 384 W)
    batch_size = 1
    image_pixels = torch.randn(batch_size, 3, 2, 384, 384)

    output = model(image_pixels)
    print(f"Input pixel shape: {image_pixels.shape}")
    print(f"Output shape: {output.shape}")
    # Expected:
    # num_height_patches, num_width_patches each 384/16 = 24
    # 24*24 = 576 patches. time=2 frames, temporal_patch_size=2 so, 2/2=1 temporal frame.
    # Total number of patches = 1 * 576. Final merge 2x2, 576/4 = 144 patches/tokens
    print(f"Expected merged patches: {24 * 24 // 4}")
