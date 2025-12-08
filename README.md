# LLM Quest

**LLM Architectures, techniques, and research papers for experimentation and learning — from scratch.**


<!-- TOC -->

- [Content](#content)
    - [Architectures](#architectures)
    - [Mixture of Experts MoE](#mixture-of-experts-moe)
    - [Alignment & Reasoning](#alignment--reasoning)
    - [Multimodal](#multimodal)
    - [Fine-tuning SFT](#fine-tuning-sft)
    - [Other Model-Agnostic Techniques and Papers](#other-model-agnostic-techniques-and-papers)
- [Acknowledgements](#acknowledgements)

<!-- /TOC -->


&nbsp;

## Latest 3 updates

- Qwen SAPO (Soft Adaptive Policy Optimization) loss implementation
- Moonshot.ai's QK-Clip compatibility with Grouped attention variants (GQA, MQA)
- Revisiting Reinforcement Pretraining (RPT) with a more robust Qwen3-0.6B

&nbsp;

## Content

> *More details are available in each subfolder's `README.md`*

&nbsp;

### Architectures

|        | Key Components |
|:-------|:---------------|
| **GPT-2**\* | • MHA<br>• LayerNorm<br>• FFN<br>• GeLU<br>• KVCache |
| **GPT to Llama 3.2**\* | • GQA<br>• RoPE + YaRN<br>• RMS Norm<br>• SwiGLU |
| **Llama 3.2 to DeepSeek V3/R1** | • MLA<br>• MTP<br>• DeepSeek MoE |
| **Llama 3.2 to Gemma 3** *(text-only)* | • GeGLU<br>• Local/Global attention<br>• SWA<br>• QK norm<br> • Pre+Post RMSNorm<br>• Logit softcapping (*Gemma 2*) |
| **Qwen3** *(dense and MoE)* | — |
| **Qwen3-Next** | • Gated DeltaNet<br>• Gated Attention<br>• Zero-Centered RMSNorm<br>• Weighted shared expert<br>• Partial RoPE |

&nbsp;

### Mixture of Experts (MoE)

| Variant | Notes |
|:--------|:------------|
| **Sparse MoE** | Classic auxiliary loss + z router loss |
| **DeepSeek MoE** | Fine-grained + shared expert isolation + auxiliary loss-free load balancing |

&nbsp;

### Alignment & Reasoning

| Method | Notes |
|:-------|:------|
| **DPO**\* | With cDPO for noisy labels, step by step |
| **RLHF with GRPO** | including variants: Dr. GRPO, DAPO, GSPO, SAPO |
| **RLVR with GRPO** | —  |
| **Qwen GSPO** | Transition from GRPO implementation |
| **Reinforcement Pretraining (RPT)** | — |


&nbsp;

### Multimodal

|        | Key Components |
|:-------|:---------------|
| **Part 1: GPT to ViT** | • Image patches + learnable CLS token + positional encoding<br>• Full Attention<br>• Classification head |
| **Part 2: VLM** | • ViT-LLM adapter (multimodal alignment/fine-tuning)<br>• Early fusion (image + text embeddings) |


&nbsp;

### Fine-tuning (SFT)

| Type | Method |
|:-----|:-------|
| **Classifier** | Hidden state retrieval for the last real token |
| **Instruction**\* | — |


&nbsp;

### Other Model-Agnostic Techniques and Papers

|           | Notes |
|:----------|:------------|
| **QK-Clip** | Query-Key clipping (naive & per head + GQA compatible) from [Moonshot.ai](https://www.moonshot.ai/)'s MuonClip and experimental "Magnitude" variant. |
| **Speculative Decoding** | Google's original version |
| **Dynamic Tanh** | Normalization-free alternative to RMSNorm/LayerNorm ([Zhu et al., 2025](https://arxiv.org/abs/2503.10622)) |
| **RoPE + YaRN** | NTK-aware + by-part/wavelength scaling |
| **LoRA**\* | — |
| **Number Token Loss** | Regression-like loss on number tokens — Wasserstein Distance variant ([Zausinger et al., 2025](https://arxiv.org/abs/2411.02083)) |
| **generate.py** | common sampling functions: temperature, top-k, top-p, min-p |
| **experimental** | — |

&nbsp;

**\*** : Already covered by @rasbt; my code is similar.

<sup>1</sup> : The original GPT-2 implementation only included causal masks, not attention masks. (In OpenAI's code,
causal masks are called ["attention
mask"](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L58C1-L58C38), which
can be confusing) 


&nbsp;

## Acknowledgements

Most notably, the Open-source community, without whom none of this would have been possible.  
Whether academia, top AI labs or independent researchers, I am grateful for their shared knowledge and research.  

Research papers used in the repo are always cited and linked in the relevant readmes or code comments.

Special mention to [@rasbt](https://github.com/rasbt) for the [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) book/repo,
  which made me kickstart this repo and became a base for verbose re-implementations of various research papers.