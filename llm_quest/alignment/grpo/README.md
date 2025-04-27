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
  [Pipeline](#pipeline)). Can be seen as inverse policy ratio too if $\pi_{ref} = \pi_{\theta_{old}}$ <sup>1</sup>

  It is not injected in the reward trajectories (per mini rewards/tokens) but computed afterward per token with the policy
  ratio during the loss calculation.

## Pipeline

### Original GRPO<sup>2, 3</sup>

3x GPT-2 models as:
- Policy model $\pi_{\theta}$:
    - SFT or Base : Pros of Base we don't get SFT bias but since it's not SFT, the model will have to learn a bit more
      by itself, likely requiring more training than an SFT'd one.

      DeepSeek used a Base DeepSeekMath 7B model.

    - $\pi_{\theta}$ is also used for $\pi_{\theta_{old}}$, not using a separate model<sup>2</sup> for sampling +
    policy ratio $\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$


- Reference model $\pi_{ref}$:
    - a copy of $\pi_{\theta}$ which stays in evaluation mode, only used for KL divergence so
      that $\pi_{\theta}$ doesn't drift away too much or tricks $r_{\phi}$. 


- Reward model $r_{\phi}$:
    - Should be pretrained on a preference (pref/rej pairs) dataset, with the output head changed to 
    $W \in \mathbb{R}^{d_{out} \times 1}$ for a single scalar reward, aimed at minimizing the Bradley-Terry loss 
    $\mathcal{L}(\phi) = - \log \sigma(R_\phi(p, r_{\text{pref}}) - R_\phi(p, r_{\text{rej}}))$.  
      $r_{\phi}$ is used to compute the mini rewards/per tokens of each $\pi_{\theta}$'s generated trajectory.

&nbsp;

*GRPO Algorithm's description from the DeepSeekMath paper*, recopied for reference as the `grpo_training_loop` function
in `grpo_engine.py`, imo, pretty much reads by itself.
> **Input:** initial policy model $\pi_{\theta_{init}}$; reward models $r_{\phi}$; task prompts $\mathcal{D}$; hyperparameters $\epsilon, \beta, \mu$
> 
> 1. policy model $\pi_{\theta} \leftarrow \pi_{\theta_{init}}$
> 2. **for** iteration = 1, ..., I **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp; reference model $\pi_{ref} \leftarrow \pi_{\theta}$
> 4. &nbsp;&nbsp;&nbsp;&nbsp; **for** step = 1, ..., M **do**
> 5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample a batch $\mathcal{D}_b$ from $\mathcal{D}$
> 6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update the old policy model $\pi_{\theta_{old}} \leftarrow \pi_{\theta}$
> 7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample G outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot | q)$ for each question $q \in \mathcal{D}_b$
> 8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute rewards $\{r_i\}_{i=1}^G$ for each sampled output $o_i$ by running $r_{\phi}$
> 9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute $\hat{A}_{i,t}$ for the t-th token of $o_i$ through group relative > advantage estimation.
> 10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** GRPO iteration = 1, ..., $\mu$ **do**
> 11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update the policy model $\pi_{\theta}$ by maximizing the GRPO objective (Equation 21)
> 12. &nbsp;&nbsp;&nbsp;&nbsp; Update $r_{\phi}$ through continuous training using a replay mechanism.
> 
> **Output:** $\pi_{\theta}$


***1*:** See [Experimental](#experimental) 

***2*:** Concerning line 6 of GRPO, it may be a misinterpretation on my part (can be modified if wrong anyway).  
For efficiency purpose, since for every new episode/batch $\pi_{\theta_{old}}$ = $\pi_{\theta}$ we can use
$\pi_{\theta}$ for old log probabilities and generating trajectories with carefully switching from `.eval()` + `torch_inference_mode()` and
`.train()` for gradient updates.  
In any case, if this holds true, DeepSeek have probably done it this way and explicitly mention $\pi_{\theta_{old}}$ for clarity.

***3***: For now the only difference is the last line (12) of the GRPO Algorithm, concerning the reward model replay
mechanism. I initially went PPO style with pretraining & freezing a reward model mentioned above, thus continuous
$r_{\phi}$ update is not implemented atm.

### Experimental

This is built on top of the ***1 & 2*** assumption with `grpo_training_loop`.  
In `grpo_training_loop_variant_experimental`, we take further risk by unanchoring $\pi_{ref}$ from
the outer loop and increasing its update frequency, keeping it much closer to $\pi_{\theta}$ by updating it every
episode. ie $(\pi_{ref}, \pi_{\theta_{old}}) \leftarrow (\pi_{\theta}, \pi_{\theta})$

This leaves us with a single model for handling $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$.  
The drawback is that $\pi_{ref}$ anchor role isn't as strong, since it's updated every episode/batch with $\pi_{\theta_{old}}$.

2x GPT-2 models as:
- Policy model for $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$

    - $\pi_{\theta}$ will be used for KL divergence + policy ratio.
    - I'm only updating $\pi_{ref}$ per episode (ie, not epoch or gradient update if `num_grad_updates > 1`).

- Reward model $r_{\phi}$:
    - Same as original GRPO minus ***3*** mentioned above.




### Additional details

- I chose Outcome Supervision (not Process Supervision) concerning the reward calculations for simplicity. Ie, 1 reward
  per trajectory but computed in a way that is compatible with process supervision:

- Using mean pooling for the full reward calculation for each trajectory.  
In case of later implementation of Process supervision, mean pooling will be naturally compatible since we retrieve all
mini rewards at each step for each trajectory. If we had used "last token only" method, it wouldn't be compatible with
process supervision.


