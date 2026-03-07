import torch
import torch.nn as nn

from llm_quest.common.rope import VisionRoPE

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
        temporal_patch_size (int): number of frames to group together
    """

    def __init__(self, img_width, img_height, num_channels, emb_dim, patch_size, temporal_patch_size):
        super().__init__()
        assert img_width % patch_size == 0, f"Image width {img_width} not divisible by patch size {patch_size}"
        assert img_height % patch_size == 0, f"Image height {img_height} not divisible by patch size {patch_size}"

        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_image_patches = (img_width * img_height) // patch_size**2  # N = HW/P^2
        # num_patches is now = self.num_image_patches * time // temporal_patch_size

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
    - GELU is approximated with tanh instead of exact GELU (using Pytorch)
    """

    def __init__(self, cfg):
        super().__init__()
        self.lin1 = nn.Linear(cfg["d_in"], cfg["hidden_dim"])
        self.lin2 = nn.Linear(cfg["hidden_dim"], cfg["d_out"])
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
        self.d_in = cfg["d_in"]
        self.num_heads = cfg["num_heads"]
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
        self.norm1 = nn.LayerNorm(cfg["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg["hidden_size"], eps=1e-6)
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
        "d_in": emb_dim,  # d_in = d_out = hidden_size
        "num_heads": 4,
    }
    cos, sin = VisionRoPE.compute_angles_2d(
        base=10000,
        head_dim=dummy_cfg["d_in"] // dummy_cfg["num_heads"],
        height_patches=num_patches_h,
        width_patches=num_patches_w,
        num_frames=num_frames,
    )

    attn = Qwen3_5VisionAttention(dummy_cfg)
    y = attn(x_shaped, cos, sin)
    print("Qwen3_5VisionAttention output shape:", y.shape)
    print(y)
