# LLM Quest: Architectures, techniques, research papers from scratch

This repo is a constant WIP.  
It was initially a "package like" version following the great
[LLM from scratch repo/book](https://github.com/rasbt/LLMs-from-scratch) structure from [@rasbt](https://github.com/rasbt).

Little by little, it served me as a base for verbose re-implementations of different architectures and research
papers. Put simply, LLM stuff that piques my interest for experiments and learning.


## Latest

- Speculative Decoding from scratch
- Reinforcement Pretraining (RPT) from scratch
- Qwen GSPO (Group Sequence Policy Optimization)
- Moonshot.ai's standalone QK-Clip technique (from MuonClip) and own Magnitude-QK-Clip variant
- RLVR Reasoning with GRPO from scratch
- Vision Transformer (ViT) from scratch
- RLHF with GRPO from scratch
- Gemma 3 architecture from scratch
- DeepSeek V3, R1 architecture from scratch
- Mixture of Experts (MoE) from scratch

<sub><sup>(did I mention "from scratch"?)</sup></sub>

&nbsp;


## Content

*More details in each subfolder's `README.md`*

 - GPT* (modified for attention masks $^1$):
    - MHA
    - Layer Norm
    - FFN
    - GeLU
    - KVCache

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

 - GPT to Vision Transformer (ViT) from scratch:
    - Image encoding: Image patches + learnable CLS token + positional encoding
    - Full Attention
    - Image Classification head
    - ViT↔LLM adapter for multimodal alignment/fine-tuning

&nbsp;
    
 - Mixture of Experts (MoE) from scratch:
    - Sparse MoE with classic auxiliary loss + z router loss
    - DeepSeek MoE variant: fine-grained + shared expert isolation + auxiliary loss-free load balancing

&nbsp;

 - GPT Fine-tuning (SFT):
    - classifier (method: retrieval of the hidden state for the last real token)
    - instruction*

&nbsp;

 - Alignment:
    - DPO* (with cDPO for noisy labels), step by step
    - RLHF with GRPO from scratch
    - RLVR Reasoning with GRPO from scratch (working but slow)
    - Qwen GSPO (transition from the GRPO implementation)

&nbsp;
    
- Common:
   - NumTokenLoss (Regression-like Loss on Number Tokens - Wasserstein Distance variant 
   [*Zausinger et al, 2025*](https://arxiv.org/abs/2411.02083))
   - QK-Clip (Query-Key clipping) from Moonshot.ai's MuonClip, alternative to logit softcapping and QK norm.
   - DyT (Dynamic Tanh, normalization free ([*Zhu et al, 2025*](https://arxiv.org/abs/2503.10622)) alternative to 
   RMSNorm, LayerNorm)
   - RoPE + YaRN (NTK aware + by-part/wavelength scaling)
   - LoRA*
   - `[prefix]_engine.py`, `engine.py` functions for training logic
   - `dataset.py` functions for preprocessing data 

&nbsp;

*\** Already covered by @rasbt, my code is similar. 
 
$^1$ The original GPT-2 implementation, at the time, didn't have attention masks but only causal masks (*in OpenAI's
code, they call the actual causal masks ["attention
mask"](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L58C1-L58C38) which
adds confusion to the terminology*).  
I implemented it mainly for SFT or RLHF related tasks to ensure the model doesn't attend from/to padding tokens + can be
used for custom losses as a mask (For CE loss, Pytorch built-in function with no_loss/ignore_index=-100 tokens is
faster).  
It's not a problem for pretraining or inference (unless batching is desired) which were the main use cases of the
original GPT-2.

&nbsp;

## potential TODOs
- non hardcoded cuda devices
- reorganize activation and normalization functions in dedicated modules
- nested TODOs
- Confusing names: model attn_mask arg (padding tokens only) and attention_mask used as loss mask for alignment
- GRPO:  
   - add process supervision
   - GRPO iterative RL variant (continuous learning of $r_{\phi}$)