import torch
import torch.nn as nn

from config import VIT_BASE_CONFIG
from llm_quest.vit.vit_transformer_block import LayerNorm, ViTTransformerBlock

# Some major differences between causal decoders and ViT here.
# First, the preprocessing step for encoding tokens vs images.
# We create a class to split our image tensors into patches of images (similar to tokens in a sequence) and project
# these patches to the embedding dimension.
#
# Concerning the positional embeddings, we don't need a table of embeddings like in GPT since our sequences are of
# fixed length here (images are of fixed WxH size and so are the patches).
#
# We also change the out projection to match our number of classes.
# Since we don't have a masked attention, we can retrieve the first learned CLS token which will be imbued with
# information of the entire image.


class PatchEmbedding(nn.Module):
    """
    Convert image tensors to sequence of patch embeddings.

    This class:
    1. Splits image into patches
    2. Linear projection of flattened patches to embedding dimension
    3. Prepends a learnable classification token

    Args:
        img_width (int): Input image width
        img_height (int): Input image height
        patch_size (int): Size of each patch (assumes square patches)
        num_channels (int): Number of input channels (RGB = 3)
        emb_dim (int): Embedding dimension
    """

    def __init__(self, img_width, img_height, patch_size, num_channels, emb_dim):
        super().__init__()
        assert img_width % patch_size == 0, f"Image width {img_width} not divisible by patch size {patch_size}"
        assert img_height % patch_size == 0, f"Image height {img_height} not divisible by patch size {patch_size}"

        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.num_patches = (img_width * img_height) // patch_size**2  # N = HW/P^2

        # If we had flattened: each patch is num_channels x patch_size x patch_size = num_channels * patch_size^2
        # Instead of manually flattening, sliding a kernel with unfold() and projecting with a linear layer,
        # we use a convolutional layer, which is basically its purpose: extract & project
        self.conv_proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,  # ie (patch_size x patch_size)
            stride=patch_size,  # sliding window step (if >= kernel size = no overlap)
            padding=0,
        )

        # learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(self, x):
        """
        Args:
            x: Input images, shape (b, num_channels, img_width, img_height)

        Returns:
            Patch embeddings, shape (b, num_patches + 1, emb_dim)
        """
        batch_size = x.shape[0]

        # convolution: extract patches & project to emb_dim
        # shape: (b, num_channels, img_width, img_height) → (b, emb_dim, num_patches_h, num_patches_w)
        x = self.conv_proj(x)

        # flattening & transposing → (b, num_patches, emb_dim)
        # Important: We can't use view() or reshape() directly because the memory layout is important here!
        # we don't want to group elements that belong to different patches or different embedding dimensions together.
        # we explicitly reorder the dimensions.
        x = x.flatten(2).transpose(1, 2)

        # expand to match batch size (for concat) and prepend classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (b, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (b, 1 + num_patches, emb_dim)

        return x


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) model adapted from GPT architecture.

    We follow the arch from the paper "An Image is Worth 16x16 Words":
    1. Patch embedding (split image into patches + linear projection) + Add classification token
    2. Add learnable positional embeddings
    3. Transformer encoder blocks (no causal masking)
    4. Layer normalization
    5. Classification head using the CLS token

    Args:
        cfg (dict): Config dictionary containing model hyperparameters

    returns:
        logits (torch.Tensor): Logits for each class, shape (b, num_classes)

        if output_hidden_states = True:
            hidden_states (torch.Tensor): final hidden, shape (b, num_patches + 1, emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()

        # Patch embedding layer (replaces token embeddings from GPT)
        self.patch_embedding = PatchEmbedding(
            img_width=cfg["img_width"],
            img_height=cfg["img_height"],
            patch_size=cfg["patch_size"],
            num_channels=cfg["num_channels"],
            emb_dim=cfg["emb_dim"],
        )
        # learnable positional embeddings (num_patches + 1 for cls token) (replaces nn.Embedding from GPT)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, cfg["emb_dim"]))
        self.dropout = nn.Dropout(cfg["drop_rate"])
        # could have used nn.Sequential but keeping nn.ModuleList in case I want to experiment with additional args
        self.transformer_blocks = nn.ModuleList([ViTTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_ln = LayerNorm(cfg["emb_dim"])

        # classification head (replaces vocab_size out projection from GPT)
        self.classifier = nn.Linear(cfg["emb_dim"], cfg["num_classes"])

    def forward(self, x, output_hidden_states=False):
        """
        Args:
            x: Input images of shape (b, num_channels, img_width, img_height)
            output_hidden_states (bool): Whether to return final hidden states only (no logits)

        Returns:
            Logits of shape (b, num_classes)
        """
        # convert images to patch embeddings and add positional embeddings
        x = self.patch_embedding(x)  # (b, num_patches + 1, emb_dim)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_ln(x)

        if output_hidden_states:
            return x  # (b, num_patches + 1, emb_dim)

        else:
            # retrieve cls token (first token) for classification
            cls_token_output = x[:, 0]  # (b, emb_dim)
            logits = self.classifier(cls_token_output)  # (b, num_classes)
            return logits


# Testing code
if __name__ == "__main__":
    torch.manual_seed(123)

    # Ex input: batch of 2 RGB images of size 224x224
    x = torch.randn(2, 3, 224, 224)

    # test patch embedding
    patch_emb = PatchEmbedding(
        img_width=224,
        img_height=224,
        patch_size=16,
        num_channels=3,
        emb_dim=768,
    )

    patch_output = patch_emb(x)
    print(f"Patch embedding output shape: {patch_output.shape}")
    print(f"Number of patches: {patch_emb.num_patches}")

    # test ViT model
    vit = ViTModel(VIT_BASE_CONFIG)

    logits = vit(x, output_hidden_states=False)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(logits)
    print(f"Model parameters: {sum(p.numel() for p in vit.parameters()):,}")
