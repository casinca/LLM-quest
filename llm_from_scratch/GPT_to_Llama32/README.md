# From GPT to Llama 3.2 architecture from scratch

- Llama papers:
  - Llama3: https://arxiv.org/abs/2407.21783
  - Llama2: https://arxiv.org/abs/2307.09288

- LayerNorm → RMSNorm
  - RMSNorm: https://arxiv.org/abs/1910.07467

- GeLU → SwiGLU (SiLU + GLU)
  - SiLU: https://arxiv.org/abs/1702.03118
  - GLU: https://arxiv.org/abs/1612.08083
  - Swish (SiLU with β=1): https://arxiv.org/abs/1710.05941

- Multi-Head Attention → Grouped-Query Attention (GQA is a middle ground between MHA and MQA)
  - GQA: https://arxiv.org/abs/2305.13245

- Absolute positional embeddings → relative + absolute with RoPE + NTK-by-part scaling (following RoPE + YaRN impl)
  - RoPE: https://arxiv.org/abs/2104.09864
  - YaRN: https://arxiv.org/abs/2309.00071
  - @rasbt impl: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-gpt-to-llama2.ipynb
  - Meta impl: https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
  - HuggingFace impl:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L83
    
  
- Global cache for RoPE params and attention mask (avoid recomputing for each transformer block)

- No dropout

- Llama hparams in config 

- Weights tying for token embs ↔ output linear layer (assuming we won't load the Meta Llama pretrained output layer 
  weights)

- adding dtype setting (not specific to Llama)