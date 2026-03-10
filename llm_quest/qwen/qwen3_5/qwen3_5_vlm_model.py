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
        This retrieves the 3D shape (t, h, w) of each separated feed / number of independent visual inputs in the vision
        prompt.
        Since multiple separated feeds (images or videos) are not supported from this, the number of feeds = 1
        So either 1 image or 1 video at a time

        Not handling variable size, so H and W should be the same for all images

        Args:
            (both formats supported)
            - 5D: (b, c, t, h, w) → image pixels - (original style, our PatchEmbedding3D)
            - 3D: (b, num_patches, features) → pre-extracted patches - (HuggingFace style, Qwen3_5VisionPatchEmbed)

        Returns:
            feeds_3d_shape: (num_feeds, 3) shape of each separated feed / number of independent visual inputs

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
