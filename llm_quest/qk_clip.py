# This is a standalone implementation of the qk clip technique from Moonshot.ai used in their MuonClip optimizer:
# https://moonshotai.github.io/Kimi-K2/ and https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
#
# QK clipping can be universally applied with any optimizer, and not specifically tied to Muon, hence a standalone
# class.
# The goal just like logit softcapping, QK norm, is to prevent attention entropy collapse/gradient explosion and
# thus training instability.
# It's also possible to implement it directly inside a PyTorch optimizer as a new method, but it's less modular imo
# QK clipping is applied after the optimizer step.

import torch


# NOTE: This implementation was made before the tech report was released, and as it's mentioned in p.3, is a naive way
# of implementing QK-clip because I'm clipping all heads of a layer if any of them were flagged for downscaling.
# The better way is, to add more granularity and clipping only the Q and K heads that were flagged.
class QKClipNaive:
    """
    Apply QK (Query-Key) naively (clip all heads of a layer) clip technique from Moonshot AI, based on MuonClip
    Optimizer: https://moonshotai.github.io/Kimi-K2/

    This method scales the query and key weights of the attention layers based on the
    maximum attention logits observed in each layer.
    It's designed to prevent attention logits from becoming excessively large, which can lead to numerical
    instability/training issues.

    Args:
        clip_threshold (float): The threshold(t) in the formula for clipping the attention logits.
        alpha (float): The alpha(α) exponent in the formula for the scaling eta(η) factor.
                        Default to 0.5 = makes scaling equally balanced for both Q and K.
                        > 0.5 = increased/reduced scaling on K/Q and inversely if < 0.5.
    """

    def __init__(self, clip_threshold, alpha=0.5):
        self.clip_threshold = clip_threshold
        self.alpha = alpha
        self.cached_qk_layers = None  # caching references to avoid recomputation

    @torch.no_grad()
    def __call__(self, model, max_attention_logits):
        """
        Applies QK-clipping to the model's Query and Key weights if their corresponding attention logits exceed the
        threshold.

        Args:
            model (torch.nn.Module): The LLM model to retrieve Q & K weights from.
            max_attention_logits (list[float]): A list containing the maximum attention logit observed in each layer.

        Modifies the model's weights in-place.
        """
        if self.cached_qk_layers is None:
            self.cached_qk_layers = [
                (model.trf_blocks[i].att.w_queries.weight, model.trf_blocks[i].att.w_keys.weight)
                for i in range(len(model.trf_blocks))
            ]

        max_attn_logit_tensor = torch.stack(max_attention_logits)
        eta_factors = torch.clamp(self.clip_threshold / max_attn_logit_tensor, max=1.0)

        # only iterating over layers with eta factor < 1 for efficiency
        eta_mask = eta_factors < 1.0
        if eta_mask.any():
            layers_to_clip = eta_mask.nonzero(as_tuple=True)[0]

            # precompute scaling factors for Q and K
            eta_values = eta_factors[layers_to_clip]
            query_scales = eta_values**self.alpha
            key_scales = eta_values ** (1 - self.alpha)

            for i, q_scale, k_scale in zip(layers_to_clip, query_scales, key_scales):
                query_weights, key_weights = self.cached_qk_layers[i]
                query_weights *= q_scale
                key_weights *= k_scale


# TODO
class QKClipMHA:
    pass
