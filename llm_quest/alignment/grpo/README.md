# GRPO from scratch on preference tuning

TODO

## GRPO

GRPO is a policy gradient method introduced in the DeepSeekMath paper: https://arxiv.org/abs/2402.03300. DeepSeek is
mixing PPO with REINFORCE (RLOO version), by keeping the same loss function as PPO with clipped surrogate objective
and a few changes:

- The advantages ${A}_{i}$ aren't calculated based on an AC method like GAE (Multistep TD variant) but much simpler as 
$A_i = \frac{r_i - \text{mean}(r_1,  r_2, \dots, r_t)}{\text{std}(r_1, r_2, \dots, r_t)}$ with $r_i$ reward of the 
$i$-th trajectory. As we can see there is no Critic trying to learn a state-value function $V(s)$.  
This is basically (for people familiar with Quant finance) the z-score of each reward across sampled trajectories.

- The KL divergence is estimated with [(Schulman, 2020) unbiased estimator](http://joschu.net/blog/kl-approx.html)

  $$𝔻_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - \log \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - 1$$

  Probability ratios for each/ $t$-th token of each/  $i$-th trajectory given the prompt $x$ and
  previous $t$ tokens in that trajectory, between the reference $\pi_{ref}$ and policy model $\pi_{\theta}$ (more details in
  [Pipeline](#pipeline)). Can be seen as inverse policy ratio too with $\pi_{ref} = \pi_{\theta_{old}}$.  

  It is not injected in the reward trajectories (per mini rewards/tokens) but computed afterward per token with the policy
  ratio during the loss calculation.

## Pipeline

3x GPT-2 models as:
- Policy model $\pi_{\theta}$:
    - SFT or Base : Pros of Base we don't get SFT bias but since it's not SFT, the model will have to learn a bit more
      by itself, likely requiring more training than an SFT'd one.

        DeepSeek used a Base DeepSeekMath 7B model.

- Reference model $\pi_{ref}$ / $\pi_{\theta_{old}}$:
    - a copy of $\pi_{\theta}$ which stays in evaluation mode, only used for KL divergence + policy ratio 
    $\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$ so
      that $\pi_{\theta}$ doesn't drift away too much or tricks $r_{\phi}$. 
    - I'm only updating $\pi_{ref}$ per episode (ie, not epoch or gradient update if `num_grad_updates > 1`), to keep it
      close enough to $\pi_{\theta}$.

- Reward model $r_{\phi}$:
    - Pretrained on a preference (pref/rej pairs) dataset, with the output head changed to 
    $W \in \mathbb{R}^{d_{out} \times 1}$ for a single scalar reward, aimed at minimizing the Bradley-Terry loss 
    $\mathcal{L}(\phi) = - \log \sigma(R_\phi(p, r_{\text{pref}}) - R_\phi(p, r_{\text{rej}}))$.  
      $r_{\phi}$ is used to compute the mini rewards/per tokens of each $\pi_{\theta}$'s generated trajectory.


TODO

### Additional details

- I chose Outcome Supervision (not Process Supervision) concerning the reward calculations for simplicity. Ie, 1 reward
  per trajectory but computed in a way that is compatible with process supervision:

- Using mean pooling for the full reward calculation for each trajectory.  
In case of later implementation of Process supervision, mean pooling will be naturally compatible since we retrieve all
mini rewards at each step for each trajectory. If we had used "last token only" method, it wouldn't be compatible with
process supervision.


