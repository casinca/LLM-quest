# Qwen3 from scratch

The good thing concerning [Alibaba's Qwen3](https://arxiv.org/abs/2505.09388) is that it's build on top of previous known open-source architectures that are
already implemented in the repo *(all the already seen research papers are linked in their dedicated implementations)*.

They chose to use:
- GQA (Grouped-Query Attention) used in Meta Llama
- Dense variants: use FFNs with the popular SwiGLU activation function
- Sparse variants: DeepSeek MoE for the fine-grained experts (but without shared experts isolation)
- QK-Norm (Query-Key normalization) in the attention layer, used in Gemma 3 series

Concerning the loss, they do not use the classic Load balancing loss (implemented in the MoE folder) but a variant
optimized for distributed training 
[global-batch load balancing loss, (Qiu et al., ACL 2025)](https://arxiv.org/abs/2501.11873)

TODO


