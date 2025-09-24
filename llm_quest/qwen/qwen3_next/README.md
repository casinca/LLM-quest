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
- MTP no real info
- they didn't mention but they do use shared expert isolation here for MoEs unlike Qwen3
- Mention that Qwen3-Next is the first model to have a fully gated architecture. We have usual residual connections with
  shared experts for MoE but also the 2 attention layers are gated. Gates in every component of the architecture.

- check info "Adopt the output gating mechanism from our prior work to reduce low-rank issues in attention."

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

