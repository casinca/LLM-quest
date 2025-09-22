# Qwen3 from scratch

The good thing concerning [Alibaba's Qwen3](https://arxiv.org/abs/2505.09388) is that it's built on top of previous known open-source architectures that are
already implemented in the repo.  
*(all the referenced research papers are already linked in the dedicated implementation
folders).*

They chose to use:
- GQA (Grouped-Query Attention) used in Meta Llama
- Dense variants: use gated FFNs with the popular SwiGLU activation function
- Sparse variants: DeepSeek MoE for the fine-grained experts (but without shared experts isolation)
- QK-Norm (Query-Key normalization) in the attention layer, used in Gemma 3 series

Concerning the MoE loss: They do not use the classic Load balancing loss (implemented in the MoE folder) but a variant
optimized for distributed training 
[global-batch load balancing loss, (Qiu et al., ACL 2025)](https://arxiv.org/abs/2501.11873).  
Note that this variant is reduced to the classic LBL in non-distributed training.

