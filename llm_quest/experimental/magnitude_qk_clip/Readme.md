# Magnitude QK-Clip

## QK-Clip main goal

The original QK-Clip based from Moonshot AI's MuonClip Optimizer:
https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf is triggered when the largest positive attention logits
in each head is larger than the defined treshold $\tau$: 

$$ S^h_{\text{max}} = \max\left(\tau, \frac{1}{\sqrt{d}} \max_{X \in B} \max_{i,j} Q^h_i K^{hT}_j\right) $$

and the scaling factor:
$$\gamma_h = \min\left(1, \frac{\tau}{S^h_{\text{max}}}\right)$$


The reason they are hunting for the largest positive attention logits is because instability in this scenario comes from
the growth of positive values, with the dot product from $Q^{h}K^{hT}$


That growth is uncapped/unbounded and if it leads to better loss, gradients will keep asking for increase, the cycle
will end in attention entropy collapse and/or `Inf` attention values contaminating the weights when the precision overflows
and ruining the training. So it makes sense that Moonshot.ai primal interest was to clip based on the largest positive
attention logits.

## The Magnitude variant

The goal of the variant is to keep the original and most important benefits of QK-Clip, by clipping based on the largest
positive attention logits, but also taking into account smallest negative attention logits.  
Hence, a symmetric clipping solely based on the largest magnitude of the attention logits in each head.  

So now we are finding the largest magnitude of the attention logits in each head:
$$ S^h_{\text{max}} = \frac{1}{\sqrt{d}} \max_{X \in B} \max_{i,j} \left|Q^h_i K^{hT}_j\right| $$

and the scaling factor:
$$\gamma_h = \min\left(1, \frac{\tau}{|S^h_{\text{max}}|}\right)$$

*(keeping $S^h_{\text{max}}$ as absolute in the $\gamma_h$ calc to make the code compatible with the original
QK-Clip. In the case we only pass max positive attention logits, an edge case where a small negative value might end up
being the max, will lead to a large negative $\gamma_h$ scaling)*

So why also chasing the smallest negative attention logits, if instability comes from the growth of positive values?

## Motivation

Clipping smallest negative attention logits isn't helping for explosion, it's not for the same goal. Negative logits in the exponential function of
the softmax tend to 0 the smaller/more negative they are. The gradient will vanish naturally being smaller and smaller.
Therefore QK weights for the dot product in that direction will consequently stabilize itself too.  

And since positive values are unbounded, the absolute value $|S^h_{\text{max}}|$ will, at some point, only come from the
largest positive attention logits, so we end up naturally with the classic QK-Clip for the rest of the training.

Having negative logits is desirable during training, to filter out the noise from the attention matrix and
focus on the most important tokens for the given context. It's not about nerfing that behavior.

The untested motivation/goal of the variant is targeted for early training as some sort of regularization, preventing the model
from being overconfident too early that some tokens are irrelevant for the given context. Keeping a bit of entropy in
the attention distribution in early training. That's about it.
