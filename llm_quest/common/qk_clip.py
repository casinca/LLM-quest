# This is a standalone implementation of the qk clip technique from Moonshot.ai used in their MuonClip optimizer:
# https://moonshotai.github.io/Kimi-K2/ and https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf
#
# QK-Clip can be universally applied with any optimizer, and not specifically tied to Muon (even though this specific
# optimizer benefits from it a lot due to its update mechanism), hence a standalone class.
#
# The goal of QK-Clip is to prevent the uncontrolled growth of QK weights, which lead to exploding attention logits
# and then numerical instability/training issues. But it also helps temporarily, for early training when QK-clip is
# applied/active, with attention entropy collapse indirectly (like logit softcapping and QK norm).
#
# QK-Clip is applied after the optimizer step.
# Max Attention logits have to be retrieved from the attention class in any case.

import torch


# NOTE: The first implementation was made before the tech report was released, and as it's mentioned in p.3, is a naive
# way of implementing QK-clip because I'm clipping all heads of a layer if any of them were flagged for downscaling.
# The better granular approach is to clip only the Q and K heads that were flagged, as in the 2nd implementation.
class QKClipNaive:
    """
    Apply QK-clip (Query-Key) technique from Moonshot AI naively (clips all heads of a layer), based on MuonClip
    Optimizer: https://moonshotai.github.io/Kimi-K2/

    It's designed to prevent attention logits from becoming excessively large by rescaling the Q and K weights of the
    attention layers, and avoid numerical instability/training issues.

    Args:
        clip_threshold (float): The threshold(τ) in the formula for clipping the attention logits.
        alpha (float): The alpha(α) exponent in the formula for scaling eta(η) factor.
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
        Applies QK-clip to the model's Query and Key weights if their corresponding attention logits exceed the
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


class QKClipMHA:
    """
    Standalone class to apply the per head QK-clip (Query-Key) technique from Moonshot AI, based on MuonClip
    Optimizer: https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf

    QK-clip is designed to prevent attention logits from becoming excessively large by rescaling the Q and K weights for
    each attention head, and avoid numerical instability/training issues.

    Args:
        clip_threshold (float): The threshold(τ) in the formula for clipping the attention logits.
        alpha (float): The alpha(α) exponent in the formula for scaling gamma(γ) factor.
                        Default to 0.5 = makes scaling equally balanced for both Q and K.
                        > 0.5 = increased/reduced scaling on K/Q and inversely if < 0.5.
    """

    def __init__(self, clip_threshold, alpha=0.5):
        self.clip_threshold = clip_threshold
        self.alpha = alpha
        self.cached_qk_layers = None  # caching references to avoid recomputation
        self.num_heads = None
        self.head_dim = None

    def _init_cache(self, model):
        if self.cached_qk_layers is None:
            self.cached_qk_layers = [
                (model.trf_blocks[i].att.w_queries.weight, model.trf_blocks[i].att.w_keys.weight)
                for i in range(len(model.trf_blocks))
            ]
            self.num_heads = model.trf_blocks[0].att.num_heads
            hidden_dim = model.trf_blocks[0].att.w_queries.weight.shape[0]  # hidden_dim = d_out
            self.head_dim = hidden_dim // self.num_heads

    @torch.no_grad()
    def __call__(self, model, max_attn_logits_per_layer):
        """
        Scales the query and key weights, per head for all layers, if their corresponding attention
        logits exceed the threshold.

        Args:
            model (torch.nn.Module): The LLM model to retrieve Q & K weights from.
            max_attn_logits_per_layer (list[torch.Tensor]): A list containing 1D (num_heads,) tensors: the maximum
            attention logits of each head in that i-th layer.

        Modifies the model's weights in-place.
        """
        self._init_cache(model)

        for i, max_attn_logits_per_head in enumerate(max_attn_logits_per_layer):
            needs_clipping = max_attn_logits_per_head > self.clip_threshold

            if not needs_clipping.any():
                continue

            gamma_factors_per_head = torch.where(  # shape (num_heads,)
                needs_clipping,
                self.clip_threshold / max_attn_logits_per_head,
                1.0,
            )

            query_weights, key_weights = self.cached_qk_layers[i]
            # reshaping Q and K weights to get num_heads matrices/weights of shape (head_dim, hidden_dim) for vect mult
            query_reshaped = query_weights.view(self.num_heads, self.head_dim, -1)
            key_reshaped = key_weights.view(self.num_heads, self.head_dim, -1)
            # applying alpha exponent to gamma factors and reshaping to (num_heads, 1, 1) for vectorized multiplication
            query_scales = (gamma_factors_per_head**self.alpha).view(self.num_heads, 1, 1)
            key_scales = (gamma_factors_per_head ** (1 - self.alpha)).view(self.num_heads, 1, 1)

            query_reshaped *= query_scales
            key_reshaped *= key_scales


# This variant is faster than QKClipMHA when there's clipping, but slower when there's no clipping.
# When only max positive attn logits are passed, this variant has the same effect as QK-ClipMHA.
# see motivation: llm_quest/experimental/magnitude_qk_clip/Readme.md
class MagnitudeQKClipMHA:
    """
    Standalone class to apply per head Magnitude-QK-Clip (Query-Key) variant inspired from the original
    QK-Clip of Moonshot AI's MuonClip Optimizer: https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf

    This variant downscales, per head, QK weights based on if the largest magnitude of the attention logits exceed the
    given threshold. (can accept both positive and negative logits)

    Args:
        clip_threshold (float): The threshold(τ) in the formula for clipping the attention logits.
        alpha (float): The alpha(α) exponent in the formula for scaling gamma(γ) factor.
                        Default to 0.5 = makes scaling equally balanced for both Q and K.
                        > 0.5 = increased/reduced scaling on K/Q and inversely if < 0.5.
    """

    def __init__(self, clip_threshold, alpha=0.5):
        self.clip_threshold = clip_threshold
        self.alpha = alpha
        self.cached_qk_layers = None  # caching references to avoid recomputation
        self.num_heads = None
        self.head_dim = None

    def _init_cache(self, model):
        if self.cached_qk_layers is None:
            self.cached_qk_layers = [
                (model.trf_blocks[i].att.w_queries.weight, model.trf_blocks[i].att.w_keys.weight)
                for i in range(len(model.trf_blocks))
            ]
            self.num_heads = model.trf_blocks[0].att.num_heads
            hidden_dim = model.trf_blocks[0].att.w_queries.weight.shape[0]  # hidden_dim = d_out
            self.head_dim = hidden_dim // self.num_heads

    @torch.no_grad()
    def __call__(self, model, max_attn_logits_per_layer):
        """
        Scales the query and key weights, per head for all layers, if the magnitude of their corresponding attention
        logits exceeds the threshold.

        Args:
            model (torch.nn.Module): The LLM model to retrieve Q & K weights from.
            max_attn_logits_per_layer (list[torch.Tensor]): A list containing 1D (num_heads,) tensors: the maximum
            attention logits of each head in that i-th layer.

        Modifies the model's weights in-place.
        """
        self._init_cache(model)

        for i, max_attn_logits_per_head in enumerate(max_attn_logits_per_layer):
            gamma_factors_per_head = torch.clamp(self.clip_threshold / torch.abs(max_attn_logits_per_head), max=1.0)

            if (gamma_factors_per_head >= 1.0).all():
                continue

            query_weights, key_weights = self.cached_qk_layers[i]
            # reshaping Q and K weights to get num_heads matrices/weights of shape (head_dim, hidden_dim) for vect mult
            query_reshaped = query_weights.view(self.num_heads, self.head_dim, -1)
            key_reshaped = key_weights.view(self.num_heads, self.head_dim, -1)
            # applying alpha exponent to gamma factors and reshaping to (num_heads, 1, 1) for vectorized mult
            query_scales = (gamma_factors_per_head**self.alpha).view(self.num_heads, 1, 1)
            key_scales = (gamma_factors_per_head ** (1 - self.alpha)).view(self.num_heads, 1, 1)

            query_reshaped *= query_scales
            key_reshaped *= key_scales


# In the per head case, we don't retrieve the max attn logit in the MHA class as:
# self.max_attn_logit = torch.max(scaled_att_scores.detach()) with scaled scores shape (b, n_heads, seq_len, seq_len)
# but:
# self.max_attn_logit = torch.amax(scaled_att_scores.detach(), dim=(0,2,3))
# ie: finding the max attn logit for each head in a given layer
