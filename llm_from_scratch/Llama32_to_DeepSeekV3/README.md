# From Llama 3.2 to DeepSeek V3 (or R1) architecture from scratch

This is an attempt at reimplementing the **training model<sup>1</sup>** of DeepSeekV3 based on below-mentioned research papers only. No engineering
optimizations are applied, which could deviate from what's being described in the papers. Any divergence (except
hyperparameters and paragraphs involving parallelism) would be unintentional and an error on my part.

The goal is to closely match the architecture as a base for research and experimental purposes while keeping the code
readable/educational.


- Grouped-Query Attention → Multi-Head Latent Attention (MLA)
  - MLA (DeepSeek V2): https://arxiv.org/abs/2405.04434

- Multi Token Prediction (MTP):
    - MTP sequential (DeepSeek V3): https://arxiv.org/abs/2412.19437v2
    - MTP parallel (Meta inspiration): https://arxiv.org/abs/2404.19737
    
- DeepSeek MoE (fine-grained w/ shared expert isolation + aux loss free) for layers ≥ 3, (first 3 are dense FFNs)
    - DeepSeekMoE: https://arxiv.org/abs/2401.06066
    - Auxiliary loss free load balancing: https://arxiv.org/abs/2408.15664

    **Note:** DeepSeek MoE code is in the [MoE from scratch folder](../MoE/) along classic sparse MoE

- DeepSeek hparams in config

&nbsp;


<sup>**1**</sup> The DeepSeek models, like many, are inference and open-weights models, which is not the same as the base model that
was used for pretraining. You can, for instance, notice that Multi Token Prediction is not present in the DeepSeek code
because:  
- the intrinsic nature of MTP is tricky to use for inference  
- they explicitly mentioned they didn't use
it for inference (although parallel MTP has been tested by Meta), mainly used to improve the training perfs.

Thankfully, the DeepSeek team, as masters of their craft, have thoroughly explained their approach to make it
reproducible.