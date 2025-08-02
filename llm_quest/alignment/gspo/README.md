# Qwen GSPO (Group Sequence Policy Optimization) from scratch

[Qwen's GSPO](https://www.arxiv.org/abs/2507.18071), which helped scale RL in their recent very competitive Qwen3
models, is pretty similar to DeepSeek's GRPO. We can say it's GRPO with primarily a single difference.  
The policy ratio (or importance ratio) is changed to being less granular. Instead of getting per tokens
(log)probs, and computing the ratio per token like:
$w_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})}$

we compute the ratio per sequence $s_i(\theta) =\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}$, with the prob
of a sequence as the product of tokens' probs $\prod_{t=1}^{|y_i|} \pi_{\theta}(y_{i,t}|x, y_{i,<t})$ or
for the code implementation, in logspace, the sum of logprobs $\sum_{t=1}^{|y_i|} \log \pi_{\theta}(y_{i,t}|x,y_{i,<t})$.  
(we still need to compute the logprobs per token, just like GRPO, in order to get the ratio per
sequence/trajectory.)

So now each $\pi_{\theta}$ / $\pi_{\theta_{\text{old}}}$ logprob represents a single trajectory from the group $\lbrace
y_i \rbrace_{i=1}^G$.  
It also matches the same shape as the Advantages $A_i$ in outcome supervision GRPO, ie we have a single reward per
trajectory and thus advantage. This is the main reason for the switch, as they said: *"the unit of optimization objective should match
the unit of reward"*, the policy ratio and the (PPO) clipping are now applied per sequence (and not per token
anymore).

But in order to avoid bias of short vs long sequences, they normalize the length by taking the geometric mean:

$s_i(\theta) = \left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)^{\frac{1}{|y_i|}} =
\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$

or again in logspace for the code implementation:

$\log s_i(\theta) = \log\left[\left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)^{\frac{1}{|y_i|}}\right]$

$= \frac{1}{|y_i|}\log\left(\prod_{t=1}^{|y_i|}
\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$ (since $\log(a^b) = b
\cdot \log(a)$ )

$= \frac{1}{|y_i|} \cdot \sum_{t=1}^{|y_i|}
\log\left(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$ (since
$\log(\prod_{i=1}^n a_i) = \sum_{i=1}^n \log(a_i)$ )

and taking the exponential of both sides, we end up back with their equation 7 from the paper:

$s_i(\theta) = \exp\left(\log s_i(\theta)\right) = \exp\left(\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}\log\left(\frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)\right)
=\left(\frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$ (since $e^{b \log a}= a^b$)





-mention the paper sequence likelihood (Zheng et al., 2023) https://aclanthology.org/2023.findings-acl.65/. Ie, 

-specify reason

-KL div?


The switch to GSPO also induces retweaking a less aggressive clipping $\epsilon$ hparam, where they mention using
$\epsilon$ in the ~$10^{e-4}$ range compared to the typical ~0.2 default we often see. Which makes sense in hindsight
since it's not token based anymore, less variance and a different order of magnitude.

Changes from the GRPO code:
