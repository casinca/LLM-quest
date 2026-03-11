# Final Multimodal Qwen3.5 Vision-Language Model (VLM)
#
# This is the Main model that combines: vision encoder/ViT (Qwen3_5VisionModel) with the text model (Qwen3_5TextModel)
# for multimodal generation.
#
# high level pipeline:
# 1. Process token IDs (also contains image placeholders/special image tokens) through text encoder → text embeddings
# 2. Process image_pixels through vision encoder → vision embeddings
# 3. Replace image placeholder (special image token IDs) with vision embeddings in the input text embeddings
# 4. Precompute 3D position IDs for MRoPE
# 5. Run combined (text+vision) embeddings through text model with MRoPE


import torch
import torch.nn as nn

from llm_quest.qwen.qwen3_5.qwen3_5_text_model import Qwen3_5TextModel
from llm_quest.qwen.qwen3_5.qwen3_5_vision_model import Qwen3_5VisionModel


class Qwen3_5VLM(nn.Module):
    """
    Final (Multimodal) Qwen3.5 Vision-Language Model.

    Composed of:
    - Qwen3_5VisionModel (vision encoder): preprocessed image pixels → vision embeddings
    - Qwen3_5TextModel (text model): combined embeddings in text embedding space (text + vision) → logits

    The model supports two modes:
    - Text-only: just pass input_ids (same as using Qwen3_5TextModel directly)
    - Multimodal: pass input_ids + image_pixels

    Args:
        cfg (dict): Configuration dictionary shared by text and vision models:
            Must contain all keys required by `Qwen3_5TextModel` and `Qwen3_5VisionModel`
    """

    def __init__(self, cfg):
        super().__init__()
        self.image_token_id = cfg.get("image_token_id", 248056)
        self.cfg = cfg
        self.vision_model = Qwen3_5VisionModel(self.cfg)
        self.language_model = Qwen3_5TextModel(self.cfg)


    def get_feeds_3d_shape(self, image_pixels):
        """
        This retrieves the 3D shape (t, h, w), in term of patches, of each separated feed / number of independent visual
        inputs in the vision prompt.
        Since multiple separated feeds (images or videos) are not supported from this, the number of feeds = 1
        So either 1 image or 1 video at a time

        Not handling variable size, so H and W should be the same for all images.

        Purpose: This is used at runtime to make MRoPE aware of the 3D image patch shapes in the input.
        Very useful in case of variable sizes or mixed feeds (ex: 1x 4k image with an HD vid) but not the case here.

        Hugging Face might compute this externally from the tokenizer or processor as `image_grid_thw`.

        Args:
            (both formats supported)
            - 5D: (b, c, t, h, w) → image pixels - (original style, our PatchEmbedding3D)
            - 3D: (b, num_patches, features) → pre-extracted patches - (HuggingFace style, Qwen3_5VisionPatchEmbed)

        Returns:
            feeds_3d_shape: (num_feeds/1, 3) shape of each separated feed / number of independent visual inputs

        """
        n_height_patches = self.vision_model.n_height_patches
        n_width_patches = self.vision_model.n_width_patches
        n_spatial_patches = n_height_patches * n_width_patches

        if image_pixels.dim() == 5:
            # shape[2] is the time/temporal dimension
            n_actual_frames = image_pixels.shape[2] // self.cfg["temporal_patch_size"]

        # if input is HF like 3D (b, num_patches, features)
        else:
            # not image_pixels in this case, patches are already temporally merged by the preprocessing
            # so n_actual_frames = num_patches // n_spatial_patches
            n_actual_frames = image_pixels.shape[1] // n_spatial_patches

        return torch.tensor([[n_actual_frames, n_height_patches, n_width_patches]])  # (1, 3) 3 for t, h, w

    def forward(self, input_ids, image_pixels=None, feeds_3d_shape=None, attn_mask=None):
        """
        Forward pass for multimodal or text-only generation.

        NOTE: When an input is multimodal, it's already "prepared" (at least the way HF does it and here). The text
        sequence length doesn't increase because of vision input.
        The text input is already expanded with image placeholder/special tokens that match the number of real vision
        tokens/patches. Therefore we just replace with a mask the image placeholder/special tokens with the vision
        tokens/patches.

        Args:
            input_ids: (b, seq_len)  token IDs (including image placeholders if multimodal)
                    Image placeholders/special tokens are injected either manually or from HF processor.
            image_pixels: (batch, in_channels, time, img_h, img_w)  preprocessed image pixels (None for text-only)
            feeds_3d_shape: (num_feeds/1, 3)  3 for (t, h, w) (None for text-only)
            attn_mask: (b, seq_len)  optional attention mask

        Returns:
            logits: (b, seq_len, vocab_size)
        """
        inputs_embs = self.language_model.emb_dict(input_ids)

        # If multimodal, process images and replace image placeholders in the input embeddings by the vision embeddings
        if image_pixels is not None:
            # retrieves vision embeddings (projected to text dim)
            vision_embeds = self.vision_model(image_pixels)  # (num_merged_patches, text_emb_dim)

            image_mask = input_ids == self.image_token_id  # (b, seq_len)
            # masked_scatter replaces True positions with values from vision_embeds into inputs_embs
            # (b, seq_len, text_emb_dim) # no shape change, just replacing
            inputs_embs = inputs_embs.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embs), vision_embeds.to(inputs_embs.dtype)
            )

            # since multimodal, we also retrieve the complete 3D image shapes for reshaping correctly
            #  the 3D position IDs
            feeds_3d_shape = self.get_feeds_3d_shape(image_pixels)  # shape (T/n_actual_frame, 3)

        # compute 3D position IDs for MRoPE
        position_ids = self.compute_position_ids(input_ids, feeds_3d_shape)

        # Forward pass through the text model with the complete input embeddings (text + vision)
        logits = self.language_model(
            inputs_embs=inputs_embs,
            position_ids=position_ids,
            attn_mask=attn_mask,
        )

        return logits


# quick test
if __name__ == "__main__":
    import config

    torch.manual_seed(123)

    cfg = config.QWEN3_5_08B_CONFIG
    dummy_cfg = dict(cfg)
    dummy_cfg.update(
        {
            "n_layers": 2,
            "emb_dim": 768,
            "hidden_dim": 3072,
            "num_heads": 12,
            "rope_base": 10000,
            "spatial_merge_size": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "img_width": 384,
            "img_height": 384,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "llm_d_in": 768,
            "image_token_id": 248056,
        }
    )

    model = Qwen3_5VLM(dummy_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    text_params = sum(p.numel() for p in model.language_model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"  Vision: {vision_params:,}")
    print(f"  Text: {text_params:,}")

    batch_size = 1
    # shape (B, C, T, H, W)
    # T/time=2 frames, so a video of 2 frames of 384x384 res
    image_pixels = torch.randn(batch_size, 3, 2, 384, 384)

    # Calculate expected number of merged image tokens
    # h_patches, w_patches = 384 // 16 = 24
    # spatial_patches = 24 * 24 = 576 per frame
    # temporal_merged_frames = 2 // temporal_patch_size = 1
    # merged_tokens = 576 // (spatial_merge_size^2) = 576 // 4 = 144
    num_image_tokens = 144
    image_token_id = dummy_cfg["image_token_id"]

    # Input IDs with placeholder [5 text] + [144 image placeholders] + [5 text] = 154 total tokens
    input_ids = torch.cat(
        [
            torch.randint(0, 1000, (batch_size, 5)),  # prefix text
            torch.full((batch_size, num_image_tokens), image_token_id),  # image placeholders
            torch.randint(0, 1000, (batch_size, 5)),  # suffix text
        ],
        dim=1,
    )

    print("\nforward pass")
    logits = model(input_ids, image_pixels=image_pixels)
    print(f"Logits shape: {logits.shape}")  # expected: (batch, 154, vocab_size)
