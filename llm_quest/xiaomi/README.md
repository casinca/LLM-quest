# Xiaomi MiMo-V2-Flash architecture from scratch



- no shared experts in MoE
- SWA and GA have different RoPE bases
- SWA and GA have different number of KV groups
- V head dim is decoupled from QK head dim
- attention sink is only applied for SWA layers
- using partial RoPE (rotating first 33% of the head dim)
- 3 different dtypes, FP32 for MoE router, BF16 for attention output, FP8 quantization for QKV?

TODO

They mention using a learnable attention sink scalar/bias added to the softmax calculation (seen in the gpt-oss paper
https://arxiv.org/abs/2508.10925).

The explanation in the paper was confusing to me initially, as they are also coupling it, in the equation (2) and (3)
below, with the max trick for exponential stability, which isn't linked to the attention sink:

$$
s_{ij} =
\frac{\exp\!\left(a_{ij} - m_i\right)}
{\exp\!\left(\text{sink} - m_i\right) + \sum_j \exp\!\left(a_{ij} - m_i\right)},
\quad
m_i = \max\!\left( \max_j a_{ij},\ \text{sink} \right).
$$

The max trick is used to avoid overflows (has nothing to do with the math) in the exponential function by subtracting
the maximum value from the logits, shifting/rescaling the values to be in the range of (-inf, 0]. This is similar to the
LSE trick. There's no need to do this manually, as the softmax function in PyTorch does it automatically.  
*Note: for precision below fp32 and in `gpt-oss` HF impl, [@ArthurZucker](https://github.com/ArthurZucker) is explicitly
doing it manually before the softmax #TODO insert lines, so maybe it was related to this when Xiaomi implemented it.*

To get back to the learnable bias, we could write the softmax for a token as:
$$
\text{Score} = \frac{e^{\text{attention}}}{\sum e^{\text{attention}} + e^{\text{sink}}}
$$

The sink is just an added learned attention score $a_{ij}$ that simulates a "fake token" which is used to dump attention to, when
the model doesn't need to attend to any other real tokens.

The goal is to remove the bias of potential real tokens (especially the first one) used as attention sinks and preserve accurate attention scores because softmax can
be seen as a competition between tokens (it has to sum to 1). This is a similar reason why DeepSeek uses sigmoid for
weighting experts in their MoE instead of the usual softmax btw

For example with 2 tokens, if the model doesn't want to pay attention to any, we can't just have [0.1, 0.1] probs, it
will be normalized to [0.5, 0.5].  
The model could even use the first token as a sink and dump attention to it [0.8, 0.2], which subsequently shift and
bias the whole attention distribution. That's why it's called **"attention sink"**.
This is a known problem, already mentioned in the Qwen3-Next `Readme.md` and documentented by Qwen in their [Gated
Attention paper](https://arxiv.org/abs/2505.06708). # TODO insert other papers from attention sink research.

