import torch
import torch.nn as nn

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


if __name__ == "__main__":
    x = torch.randn(1, 3, 8, 32, 32)
    patch_embedding = PatchEmbedding3D(
        img_width=32,
        img_height=32,
        num_channels=3,
        emb_dim=128,
        patch_size=8,
        temporal_patch_size=2,
    )
    print(patch_embedding(x).shape)  # (1, 64, 128)
    # time is 8/temporal_patch_size = 4
    # num_patches h and w (since square) is 32/patch_size = 4 for both,
    # emb_dim is 128
    # so (1, 4*4*4, 128) = (1, 64, 128)
