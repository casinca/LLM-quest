# Xiaomi MiMo-V2-Flash architecture from scratch

[MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf) is a hybrid attention (SWA+GA) model
with DeepSeek V3 inspirations for their MoE and lightweight MTP (Multi-Token Prediction) modules. These modules are not
just for training but also repurposed for inference as draft heads for speculative decoding.

<p align="center">
<img src="image.png" alt="alt text" width="600">
</p>

Some details of the architecture:
- First layer is dense GA+FFN (not MoE)
- No shared experts in MoE
- MTP modules are SWA only and share their embedding+LM head with the main model (tied weights)
- SWA and GA layers have different RoPE bases
- SWA and GA layers have a different number of KV groups (GQA)
- Values head dim is decoupled from QK head dim
- Attention sink is only applied for SWA layers
- Partial RoPE (rotating first 33% of the head dim)
- 3 different dtypes, FP32 for MoE router, BF16 for attention output, FP8 quantization for QKV (not done here)

&nbsp;

They mention using a learnable attention sink scalar/bias added to the softmax calculation (seen in the OpenAI 
[gpt-oss paper](https://arxiv.org/abs/2508.10925)).

The explanation in the MiMo paper was initially confusing to me, as they couple it with the "max trick" for exponential
stability, in their equation (2) and (3) below, which isn't linked to the attention sink in the first place:

$$
s_{ij} =
\frac{\exp\!\left(a_{ij} - m_i\right)}
{\exp\!\left(\text{sink} - m_i\right) + \sum_j \exp\!\left(a_{ij} - m_i\right)},
\quad
m_i = \max\!\left( \max_j a_{ij},\ \text{sink} \right).
$$

The max trick is used to avoid overflow (has nothing to do with the math) in the exponential function by subtracting
the maximum value from the logits, which shift/rescale the values to be in the range of $(-inf, 0]$. This is similar to the
LSE trick.  
There is no need to do this manually, as the softmax function in PyTorch does it internally.  
*Note: for precisions below fp32 in `gpt-oss` HF impl, [@ArthurZucker](https://github.com/ArthurZucker) is explicitly
[doing it manually before the softmax](https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/gpt_oss/modular_gpt_oss.py#L236-L239), so maybe it was related to this when Xiaomi mentioned it.*

To get back to the learnable bias, we could write the softmax score/prob $s_{ij}$ for real tokens $i$ and $j$ as:

$$
s_{ij} = \frac{e^{a_{ij}}}{\sum_{j'} e^{a_{ij'}} + e^{\text{sink}}}
$$

The sink is just an added learned attention score in the denominator, solely for the softmax calculation, that simulates
a "fake token" which is used to dump attention to when the model doesn't need to attend to any other real tokens.  
We could see the QK matrix for 2 real tokens as: 

| Query $i$ | $j=1$ | $j=2$ | Sink Term |
| :--- | :---: | :---: | :---: |
| **Token 1** | $e^{a_{11}}$ | $0$ | $e^{\text{sink}}$ |
| **Token 2** | $e^{a_{21}}$ | $e^{a_{22}}$ | $e^{\text{sink}}$ |

- denominator row 1 sum $= e^{a_{11}} + e^{\text{sink}}$
- denominator row 2 sum $= e^{a_{21}} + e^{a_{22}} + e^{\text{sink}}$

From the formula $s_{ij}$ above, we can see that when the sum of the real tokens in the denominator is low, the sink
will dominate and artificially make all $s_{ij}$ probs for the real tokens close to 0, effectively allowing the head not
to pay attention to real tokens.  
These new real tokens scores (that do not sum up to 1 anymore) are then used for scaling the Values.

The goal is to remove the bias of potential real tokens (very often the first one) used as attention sinks and preserve
accurate attention scores (because softmax can be seen as a competition between tokens since it has to sum to 1).  
This is a similar to why DeepSeek uses a sigmoid for weighting experts in their MoE instead of the usual softmax.

For example with 2 tokens, if the model doesn't want to pay attention to any, we can't just have [0.1, 0.1] probs, it
will be normalized to [0.5, 0.5].  
The model could even use the first token as a sink and dump attention to it like [0.8, 0.2], which subsequently shift
and biases the whole attention distribution. That's why it's called **"attention sink"**.  
This is a known problem, already mentioned in the Qwen3-Next `Readme.md` here and documented in their [Gated
Attention paper](https://arxiv.org/abs/2505.06708), among others detailed [1](https://arxiv.org/abs/2410.10781), 
[2](https://arxiv.org/abs/2309.17453) papers in the literature.

&nbsp;

## Acknowledgements

Papers mentioned in one place:

- [MiMo-V2-Flash paper](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf)
- [Gated Attention paper](https://arxiv.org/abs/2505.06708)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/abs/2410.10781)
- [gpt-oss paper](https://arxiv.org/abs/2508.10925)


