# Qwen GSPO (Group Sequence Policy Optimization) from scratch

[Qwen's GSPO](https://www.arxiv.org/abs/2507.18071), which helped scale RL for their recent very competitive Qwen3
models (at the time of writing) is pretty similar to DeepSeek's GRPO. We can say it's GRPO with one primary difference.  
The policy ratio (or importance ratio) is made less granular. Instead of getting per-token (log)probs, and applying the
ratio per-token like:
$w_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})}$

we compute the ratio per-sequence $s_i(\theta) =\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$. Since the prob
of a sequence is the product of its tokens' probs $\prod_{t=1}^{|y_i|} \pi_{\theta}(y_{i,t}|x, y_{i,<t})$ or
for the code implementation, in logspace, the sum of logprobs $\sum_{t=1}^{|y_i|} \log \pi_{\theta}(y_{i,t}|x,y_{i,<t})$,
we still need to compute the logprobs per-token, just like GRPO, in order to get the ratio per
sequence/trajectory.

It now also matches the shape of the Advantages $A_i$ in outcome supervision GRPO, ie we have a single reward
per-sequence/trajectory and consequently a single advantage/seq.  
This is the main reason Qwen introduced GSPO, as they said: *"the unit of optimization objective should match
the unit of reward"*. Both the policy ratio and the (PPO) clipping are now applied per-sequence, rather than per-token.

In order to avoid bias of long vs short sequences, they normalize for length by taking the geometric mean of per-token
probs for both policies:

$s_i(\theta) = \left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)^{\frac{1}{|y_i|}} =
\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$

or again in logspace (for the code implementation, we'll simply use the arithmetic mean):

$\log s_i(\theta) = \log\left[\left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)^{\frac{1}{|y_i|}}\right]$

$= \frac{1}{|y_i|}\log\left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$ (since $\log(a^b) = b
\log (a)$ )

$= \frac{1}{|y_i|} \cdot \sum_{t=1}^{|y_i|}
\log\left(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$ (since
$\log(\prod_{i=1}^n a_i) = \sum_{i=1}^n \log(a_i)$ )

and taking the exponential of both sides, we end up back with their *equation 7* from the paper:

$s_i(\theta) = \exp\left(\log s_i(\theta)\right) = \exp\left(\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}\log\left(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)\right)
=\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$ (since $e^{b \log a}= a^b$)

Overall, GSPO is mainly an improvement targeted at MoE models training as mentioned in the paper and by @chujiezheng
[here](https://github.com/volcengine/verl/pull/2775#issuecomment-3134375131), with performance similar to GRPO for
dense models.

&nbsp;

## Why Qwen judged important to match the granularity of the policy ratio with the advantages?

In GRPO, the gradient of each token's logprob is individually scaled by its respective per-token policy ratio. However,
this granularity can be noisy (as a single sample estimate) and raises undesired variance, which in the long-term,
accumulates, and can lead to instability during training.  
Therefore, switching to a per-sequence policy ratio, now the gradients for all tokens within a sequence are weighted by
the same sequence-level policy ratio, which reduces the instability mentioned above. Ref their GSPO gradient *equation
10*.

## The more flexible Token-level Variant

In the case we want to keep the advantages $A_{i,t}$ at the token-level (like for process supervision) but still get
the benefits of GSPO with the sequence-level policy ratio, they made a smart flexible variant (by controlling the
gradient flow in *equation 14* with PyTorch's `.detach()`) that allows both advantage granularities:  
If the advantages are at the sequence-level, then it acts the same as the original GSPO, otherwise advantages will scale
per-token while still receiving per-sequence weights from the policy ratio.

&nbsp;

## Changes from the GRPO code

- Based on additional information from @chujiezheng a Qwen researcher, in
  https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306, they also mention trying not using a KL div
  constraint by setting the beta to 0.

- The switch to GSPO induces retweaking a less aggressive (PPO) clipping $\epsilon$ hparam, where they mention using
$\epsilon$ in the ~$10^{-4}$ range, specifically different values for the min/max ($3 \cdot 10^{-4}$/$4 \cdot 10^{-4}$)
mentioned in the link above compared to the typical ~0.2 default we often see.
In hindsight, it makes sense, as it's no longer token-based with less variance and a different order of magnitude.  
This decoupled PPO clipping originated from the [DAPO paper](https://arxiv.org/abs/2503.14476).

- We create a new function `log_probs_per_seq` that reuses `log_probs_per_token` from `grpo_engine.py` to compute the
  logprobs per-token and uses the `loss_mask/reward_mask` to correctly calc the mean only for the logprobs of the
  generated tokens.

    Then we can use as a drop-in replacement for the `rlhf_grpo_training_loop` or `rlvr_grpo_training_loop`:

    ```python
    # instead of [pol/old]_logprobs = log_probs_per_token(logits, inputs, loss_mask)
    seq_pol_logprobs = log_probs_per_seq(logits, inputs, loss_mask)
    seq_old_logprobs = log_probs_per_seq(logits, inputs, loss_mask)
    seq_policy_ratio = torch.exp(seq_pol_logprobs - seq_old_logprobs)
    
    # in the grad loop...
    # no need to unsqueeze the advantages anymore, both shapes are (B,)
    surr_obj_per_sequence = policy_ratio_per_sequence * advantages
    clipped_surr_obj_per_sequence = torch.clip(
        policy_ratio_per_sequence, min=1.0 - eps_clip_min, max=1.0 + eps_clip_max
    ) * advantages
    ```