# LLM Quest: Models, architectures, research papers from scratch

This repo is a constant WIP.  
It was initially a "package like" version following the great
[LLM from scratch repo/book](https://github.com/rasbt/LLMs-from-scratch) structure from [@rasbt](https://github.com/rasbt).

Little by little, it served me as a base for verbose re-implementations of different architectures, research
papers or experiments from scratch, such as: Mixture of Experts (MoE), Gemma 3, DeepSeek V3...  
Put simply, SOTA LLM stuff that piques my interest for experiments and learning.

Even though the code is mine (unless explicitly mentioned like `gpt_download.py`), anyone
familiar with the LLM from scratch repo/book should be familiar with the code here.

## Content

**More details in each subfolder's `README.md`**

 - GPT*:
    - MHA
    - Layer Norm
    - FFN
    - GeLU

&nbsp;

 - GPT to Llama 3.2 from scratch*:
    - GQA
    - RoPE + YaRN
    - RMS Norm
    - SwiGLU

&nbsp;

 - Llama 3.2 to DeepSeek V3, R1 from scratch:
    - MLA
    - MTP
    - DeepSeek MoE

&nbsp;

 - Llama 3.2 to Gemma 3 from scratch (text-only):
    - GeGLU
    - Local/Global attention
    - SWA
    - QK norm
    - Logit softcapping (Gemma 2, kept for reference)

&nbsp;
    
 - MoE:
    - Sparse MoE with classic auxiliary loss + z router loss
    - DeepSeek MoE variant: fine-grained + shared expert isolation + auxiliary loss free load balancing

&nbsp;

 - GPT Fine-tuning*:
    - classifier
    - instruct

&nbsp;

 - Alignment:
    - DPO* (w/ cDPO for noisy labels), step by step
    - GRPO from scratch

&nbsp;
    
- Common:
   - DyT (Dynamic Tanh, normalization free ([*Zhu et al, 2025*](https://arxiv.org/abs/2503.10622)) alternative to 
   RMSNorm, LayerNorm)
   - RoPE + YaRN (NTK aware + by-part/wavelength scaling)
   - LoRA*
   - `[prefix]_engine.py`, `engine.py` functions for training logic
   - `dataset.py` functions for preprocessing data 

&nbsp;

**\*** Already covered by @rasbt, my code is similar.

## potential TODO
- non hardcoded cuda devices
- vectorize MoE dispatching while keeping the code readable
- reorganize activation and normalization functions in dedicated modules
- better optim for classification: we use masks to retrieve the last valid token instead of slicing [:,-1,:]
- nested TODOs
- GRPO: 
   - add process supervision
   - add some more training metrics
   - GRPO iterative RL variant (continuous learning of $r_{\phi}$)
   - we could return the model, instead of inplace