# Qwen3-Next from scratch

TODO

https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list

- hybrid linear:SDPA(GQA) (3:1 ratio, Gated DeltaNet:Gated SDPA(GQA) with a sigmoid)
- partial RoPE: only rotate first 25% features/dimensions for each QK
- remove all RMSNorm (including QK), Use Zero-Centered RMSNorm with weight decay/L2  
  
  - Despite "L2" being the same color as ZCRMSNorm in the Qwen diagram, it has nothing to do with weight decay for
    RMSNorm. It's additional L2 specifically for QK.
  - overall weight decay for ZCRMSNorm changes nothing for the optimizer step (including norm layers by default) It could
    have been potentially highlighted in cases where we don't usually include normalization layers in the optimizer step
    akin to https://github.com/karpathy/minGPT/issues/23 not sure without paper but it seems to be an optimizer
    detail and not an architecture impl.

    In any case the ZCRMSNorm purpose is still valid with below point

- Normalize MoE gate/router weights during init, need post init function
- 
- MTP no real info
- they didn't mention but they do use shared expert unlike Qwen3:  
  BUT MoE isn't pure expert isolation (residual like) like DeepSeekMoE, but a single large shared expert variant that is
  weighted by a single scalar gate, these raw weights are normalized with a sigmoid to scale (0, 1) the shared output.  
  funny exactly part c) of my experimental weighting shared experts from april  
  https://github.com/casinca/LLM-quest/blob/master/llm_quest/experimental/weighting_shared_experts/Readme.md

- Mention that Qwen3-Next is probably the first midsize open-source model to have a FULLY gated transformer block architecture:
  - Gated classic sdpa
  - Gated linear attention
  - Shared expert aren't "gated" in the same sense, since it's more like weighted residuals and it's additive not
    multiplicative.
    
    Need to reword this properly

- Also mention that Qwen3-Next is the most complex open-source LLM architecture from a top lab, at the time of
  writing, incorportaing the very SOTA research on linear attention with their own, more classic, Gated Attention in a
  hybrid package.

- One might ask why are we not also using RoPE for GDN? The inherent way of computing linear attention
  (recurrent/sequentially) gives a natural sense of order already  
  This is also why we don't use causal mask for recurrent GDN (like we do for Gated Attention) but only use an attention
  mask/padding mask. We only ever have access to the previous state $S_{t-1}$ and the current input $x_t$ for $S_t$, not
  $S_{t+1}...$  
  For training efficiency (whole sequence at once) the convolutional layer via padding takes care of
  the causality.

- mention difference with official Qwen3-Next implementation:
  - Not using FLA with chunked GDN for simplicity (recurrent GDN only)
  - Not using fused linears, easier to read/follow along with the architecture visualization in the paper/blogpost
  - mention diff with paper equation and qwen3-next impl (S_t vs S_t^T)


- alpha is not just a scalar passed through a linear layer with projection reduced to (0,1) with a sigmoid but more
  sophisticated like Space state models are doing $\alpha_t = e^{-A \cdot \Delta t_t}$ Mamba paper eq 4


- check info "Adopt the output gating mechanism from our prior work to reduce low-rank issues in attention."
Gated attention https://arxiv.org/abs/2505.06708

## ZCRMS part
so Zero-Centered RMSNorm is not what it seems/interepreted as doing:

$$ x_{\text{RMS\_scaled}} = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} $$


$$ x_{\text{Zero\_Centered}} = x_{\text{RMS\_scaled}} - \text{mean}(x_{\text{RMS\_scaled}}) $$


but they mean in fact initializing the weights/coeff 0 centered (unlike 1s) but since we can't multiply by 0, they add
for the forward pass a 1 constant to compensate, ie $x \cdot (1+w)$

so basically it's zero centered "weights", not zero centered "RMS", they are changing the baseline to 0 instead of 1
and the model now learns to change the scale from 0 and they just shift by 1 for the correct activation scaling

The whole point was to counter abnormally large weights in QK norm, centereing weights around 0 indeed helps as a
better starting point.
But the additional main reason they are doing this is as a very smart trick to make L2 regularization work iiuc:

if they apply L2 on a classic RMSNorm it will push the coeff/weights (starting from 1) to 0, which breaks the RMSNorm forward.
But if they apply L2 on Zero-Centered weights RMSNorm, it'll still push the weights (starting from 0) to 0 (so weights are
kept low) and for the forward only they just offset by adding 1 to keep the correct RMSNorm scaling. 

But also pushing weights towards 0, intrinsically pushes the coefficients towards
$(1 + \text{weights} (\approx 0)) \approx 1$
which is what we'd want for a reasonable scaling.
range (whereas for classic RMSNorm, weights = coeff = potential explosion). All in all, making a RMSNorm with L2
possible.


## Linear attention

https://sustcsonglin.github.io/blog/2024/deltanet-1/

linear attention: https://arxiv.org/abs/2006.16236
DeltaNet improvement with parallekism: https://arxiv.org/abs/2406.06484
GDN: https://arxiv.org/abs/2412.06464
delta net: https://proceedings.mlr.press/v139/schlag21a.html
delta rule:
https://direct.mit.edu/books/edited-volume/5431/chapter-abstract/3958517/1960-Bernard-Widrow-and-Marcian-E-Hoff-Adaptive

mamba: https://arxiv.org/abs/2312.00752

venn diagram source: https://www.nature.com/articles/s42256-025-01034-6

Gated DeltaNet (GDN) is a gated variant, from Nvidia researchers, of the former linear attention (insert link) and is a
strong contender over other gated variants, with SOTA performance. DeltaNet (without gating) had already perfect scores
in "in-context" learning but also in fuzzy and noisy recall in MAD (insert link) benchmark.

Originally GDN was incorporated in hybrid architecture with SWA and/or Mamba attention blocks (like the Gated
DeltaNet-H2 model) but Qwen with Qwen3-Next opted instead to implement GDN with their own gated GQA (one of the variant
from their Gated Attention paper)

start from classic attention equation
remove softmax and we get linear attention equation
(develop equations)

but from the linear attention equation, they don't update the state matrix $S_t$ by simply adding outer kv products
$\mathbf{k}_t \mathbf{v}_t^T$ to the previous state $S_{t-1}$.  
So instead, they update the state matrix by adapting the delta rule (link paper), ie instead of adding the full outer
kv products to the previous state $S_{t-1}$, they add an error adjusted $\mathbf{k}_t \mathbf{v}_t^T$ which is also
regulated by a learning rate $\beta$.  
the whole thing is similar to classic SGD (stochastic gradient descent) update steps

(develop equation)

That is DeltaNet (link paper delta net original, not their optimized version)

On top of that DeltaNet architecture, they add a gating term (forgetting mechanism, similar to newer RNNs), which
gave birth to Gated DeltaNet.

(develop equation)

To summarize:

The evolution of linear attention, is that standard linear attention had problems with retrieval error, and the gating
mechanism from gated variants (GLA, MAMBA) were an improvement in that direction. But these gated linear variants are
still subpar with in-context learning/recall. These gated variants were in turn improved by integrating the delta rule.
We end up with current SOTA: Gated DeltaNet.






