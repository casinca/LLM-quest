# GRPO from scratch on preference tuning

Only external dependency is `tiktoken` for tokenization.  
We're not using `transformers`, we're collating (batching, padding, masking...) ourselves & using from scratch models,
starting with the reward model training before GRPO itself.

## GRPO

GRPO is a policy gradient method introduced in the DeepSeekMath paper: https://arxiv.org/abs/2402.03300. DeepSeek is
mixing PPO with REINFORCE (RLOO version), by keeping the same loss function as PPO (surrogate objective with clipping)
and a few changes:

- The advantages ${A}_{i}$ aren't calculated based on an AC technique like GAE (~Multistep TD variant) but much simpler,
as $A_i = \frac{r_i - \text{mean}(r_1,  r_2, \dots, r_t)}{\text{std}(r_1, r_2, \dots, r_t)}$ with $r_i$ the reward of
the $i$-th trajectory. As we can see, there is no Critic trying to learn a state-value function $V(s)$.  
This is basically (for people familiar with Quant finance) the z-score of each reward across sampled trajectories.

- The KL divergence is estimated with [(Schulman, 2020) unbiased estimator](http://joschu.net/blog/kl-approx.html)

  $$𝔻_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - \log \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - 1$$

  Probability ratios for the $t$-th token of each $i$-th trajectory given the prompt $x$ and
  previous $t$ tokens in that trajectory, between the reference $\pi_{ref}$ and policy model $\pi_{\theta}$ (more details in
  [Pipeline](#pipeline)).  
  Can be seen as inverse policy ratio too if $\pi_{ref} = \pi_{\theta_{old}}$ <sup>1</sup>

  The KL div is not injected in the trajectories' full reward (per mini rewards/tokens) like PPO but computed afterward per
  token with the policy ratio during the loss calculation.

## Pipeline

### Original GRPO<sup>2, 3</sup>

3x models (here for testing the logic, 3x GPT-2 from scratch based on
[`llm_quest/gpt`](/llm_quest/gpt/)) as:

- Policy model $\pi_{\theta}$:
    - SFT or Base : Advantage of Base, we don't get SFT bias, but the model will have to learn a bit more, likely
      requiring more training since it's not SFT.

      DeepSeek used a Base DeepSeekMath 7B model.

    - $\pi_{\theta}$ is also used for $\pi_{\theta_{old}}$, not using a separate model<sup>2</sup> for sampling +
    policy ratio $\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$


- Reference model $\pi_{ref}$:
    - a copy of $\pi_{\theta}$ which stays in evaluation mode, only used for KL divergence so
      that $\pi_{\theta}$ doesn't drift away too much or tricks $r_{\phi}$. 


- Reward model $r_{\phi}$:
    - We're supposedly pretraining it in `reward_model_training.py` on a preference (pref/rej pairs) dataset, with
    $r_{\phi}$'s output head changed to $W \in \mathbb{R}^{d_{out} \times 1}$ for a single scalar reward and aimed at
    minimizing the Bradley-Terry loss $\mathcal{L}(\phi) = - \log \sigma(R_\phi(p, res_{\text{pref}}) - R_\phi(p, res_{\text{rej}}))$ with $p$ for prompt.  
    $r_{\phi}$ is used to compute the mini rewards/per tokens of each generated trajectory from $\pi_{\theta}$.

&nbsp;

*GRPO Algorithm's description from the DeepSeekMath paper*, recopied for reference, imo, the `grpo_training_loop` function
in `grpo_engine.py`, pretty much reads by itself compared to the pseudocode below, so I won't go into unnecessary details.


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
> 9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute $\hat{A}_{i,t}$ for the t-th token of $o_i$ through group relative advantage estimation.
> 10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** GRPO iteration = 1, ..., $\mu$ **do**
> 11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update the policy model $\pi_{\theta}$ by maximizing the GRPO objective (Equation 21)
> 12. &nbsp;&nbsp;&nbsp;&nbsp; Update $r_{\phi}$ through continuous training using a replay mechanism.
> 
> **Output:** $\pi_{\theta}$


***1*:** *See [Experimental](#experimental)*

***2*:** *Concerning line 6 of GRPO, it may be a misinterpretation on my part (can be modified if wrong anyway).  
For efficiency purpose, since every new episode/batch $\pi_{\theta_{old}}$ = $\pi_{\theta}$ we can use
$\pi_{\theta}$ for old log probabilities and generating trajectories with carefully switching from `.eval()` + `torch_inference_mode()` and
`.train()` for gradient updates.*

***3***:  *The last line (12) of the GRPO Algorithm, DeepSeek included it for completeness. This is concerning the
optional iterative RL alternative where the reward model isn't frozen anymore during RLHF, but continuously trained
(replay mechanism). I initially went PPO style with pretraining & freezing a reward model (which is also described as
frozen in figure 4 of the paper), thus continuous* $r_{\phi}$ *update is not implemented atm.*

### Experimental (untested mix of GRPO iterative RL)

This is built on top of the ***1 & 2*** assumption from `grpo_training_loop` and the fact that for the iterative RL
variant, DeepSeek set the reference model as the policy model.  
In `grpo_training_loop_variant_experimental`, we take a risk by unanchoring $\pi_{ref}$ from
the outer loop and increasing its update frequency, keeping it much closer to $\pi_{\theta}$ by updating it every
episode. ie $(\pi_{ref}, \pi_{\theta_{old}}) \leftarrow (\pi_{\theta}, \pi_{\theta})$

The positive point, we're left with a single model for handling $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$,
giving DPO-like vibes with only 2 models in total.  
The drawback is that $\pi_{ref}$ anchor role won't be as strong and bias will increase, since it's updated every episode/batch with $\pi_{\theta_{old}}$.

2x models as:
- Policy model for $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$

    - $\pi_{\theta}$ will be used for KL divergence + policy ratio.
    - I'm only updating $\pi_{ref}$ per episode (ie, not epoch or gradient update if `num_grad_updates > 1`).

- Reward model $r_{\phi}$:
    - Same as original GRPO minus ***3*** mentioned above.




### Additional details

- I chose Outcome Supervision (not Process Supervision) concerning the reward calculations for simplicity. Ie, 1 reward
  per trajectory, but computed in a way that is compatible with process supervision:

- Using mean pooling for the full reward calculation for each trajectory.  
In case of later implementation of Process supervision, mean pooling will be naturally compatible since we retrieve all
mini rewards at each step for each trajectory. If we had used "last token only" method, it wouldn't be compatible with
process supervision.


