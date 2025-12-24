# Speculative Decoding from scratch

This is a from scratch implementation of the original speculative decoding (speculative sampling) paper from Google
[Fast Inference from Transformers via Speculative Decoding, Leviathan et al., 2023](https://arxiv.org/abs/2211.17192).  

This is the first variant among others, like [Medusa](https://sites.google.com/view/medusa-llm) with multiple token
prediction heads (similar to DeepSeek V3/R1 with their MTP modules 
[implemented here](https://github.com/casinca/LLM-quest/tree/master/llm_quest/llama3_to_deepseekv3)).

<p align="center">
  <img src="_spec_decoding_illustration_google.gif" alt="Speculative Decoding Illustration" />
</p>

<p align="center">
  <em>Speculative Decoding Illustration: Credit Google Research from the linked blog post below.</em>
</p>

&nbsp;

The underlying intuition of speculative decoding is that some tokens are easier to predict than others, therefore we
can leverage a smaller model to generate these easier tokens accurately vs wasting compute (compared to harder tokens)
on them when using a larger/SOTA model.

Without going into too many details, as it's already well explained and detailed in their blog post:
https://research.google/blog/looking-back-at-speculative-decoding/, we are speeding up inference by using 2 models.

A draft/approximation (smaller) model is used to generate `draft_max_gen` (*called* $γ$ *in the paper*) tokens per
sub-generations (`_speculative_step`) until EoS is hit or max generation `max_gen` is reached.

The predicted logits (softmaxed into PMFs) from the draft model are then compared to the logits from the
target (larger/main) model for verification/acceptance (The target model's logits for the drafted sequence are retrieved
in a single parallel forward pass, in similar fashion to the implementation of RLHF/RLVR or DPO).  

This comparison in the code is split into 2 cases, for readability, depending on hyperparameters chosen:
- Deterministic (simpler/faster): greedy sampling (temperature = 0.0) with `speculative_sampling_greedy()`
- Stochastic: speculative sampling (temperature > 0.0, optionally top-k or top-p) with `speculative_sampling()`

&nbsp;

## Also worth noting

The `draft_max_gen` ($γ$) is a very important hparam and the difference is noticeable when doing different runs with
different values in `spec_decoding_generate.py`.

If too high, it can lead to a significant slowdown (worse than simply generating with the target model and KVcache) where
we will waste compute on some drafted tokens that will end up being discarded as soon as the first rejection is hit.  
For example, if we draft 10 tokens and the 2nd token is rejected, we will have wasted compute for the 8 following tokens.

Too low, nerfed, is a waste of potential speed up and counterproductive, with too many drafting steps per `max_gen` (or
until EoS is hit).

The delta/difference of size (number of parameters) and the alignment between the target and draft models is also
important. Similar size and the overhead of doing speculative decoding will outweigh the speed-up.  
A large delta might introduce too much divergence between the draft and target model logits and therefore untimely
rejections.  
Here, we used as a simple test case: a GPT-2 Small (124M) for drafting and a GPT-2 Large (774M) for verification.

Stochastic was troublesome to properly test with matching PyTorch random seed consumption vs a classic generation,
so the results are not reliable because the outputs don't match.  
With deterministic greedy sampling, the speed-up looks correct on average, some quick benchmarks results:

```bash
with draft_max_gen = [1, 2, 4, 8]
max_gen = [50, 100, 200, 500]

Average Speedup: 1.85x (min: 0.92x, max: 2.64x)
Average Throughput: 47.3 tokens/s (min: 23.1, max: 71.2)

Different gamma values:
Gamma    Avg Speedup  Avg Tokens/s  Tests   
------------------------------------------
1        1.23         31.4          16      
2        1.67         43.1          16      
4        2.15         58.9          16      
8        1.94         52.7          16      
```



