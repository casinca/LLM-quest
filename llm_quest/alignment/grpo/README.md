# RLHF with GRPO from scratch on preference tuning

Only external dependency is `tiktoken` for tokenization.  
Not using `transformers`, we're collating (batching, padding, masking...) ourselves & only starting with models
from scratch.

See [visualization](#visualization-of-the-pipeline) for the pipeline or jump straight to some interesting
[results](#results) (or funny reward hacking [example](#reward-hacking-and-model-collapse)).

## GRPO

GRPO is a pure policy gradient method introduced in the DeepSeekMath paper: https://arxiv.org/abs/2402.03300. DeepSeek
mixes PPO with REINFORCE (similar to Baseline or RLOO version), by keeping the same loss function as PPO (surrogate objective with clipping)
and a few changes:

- The advantages ${A}_{i}$ aren't calculated based on a hybrid AC technique using GAE but much simpler,
as $A_i = \frac{r_i - \text{mean}(r_1,  r_2, \dots, r_t)}{\text{std}(r_1, r_2, \dots, r_t)}$ with $r_i$ the reward of
the $i$-th trajectory (achieved through [Outcome Supervision](#additional-details)).  
As we can see, there is no Critic trying to learn a state-value function $V(s)$ nor using Temporal Difference.  
This is basically the z-score (for people familiar with Quant finance) of each reward across sampled trajectories.

- The KL divergence is estimated with [(J. Schulman, 2020) unbiased estimator](http://joschu.net/blog/kl-approx.html):

  $$𝔻_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - \log \frac{\pi_{ref}(y_{i,t} | x, y_{i,<t})}{\pi_\theta(y_{i,t} | x, y_{i,<t})} - 1$$

  Probability ratios for the $t$-th token of each $i$-th trajectory given the prompt $x$ and
  previous $t$ tokens in that trajectory, between the reference $\pi_{ref}$ and policy model $\pi_{\theta}$ (more details in
  [Pipeline](#pipeline)).  
  Can be seen as inverse policy ratio too if $\pi_{ref} = \pi_{\theta_{old}}$ <sup>1</sup>

  The KL div is not injected in the trajectories' full reward (per mini-rewards/tokens) like PPO but computed afterward per
  token with the policy ratio during the loss calculation.
  
- No Entropy bonus $H(\pi_\theta(\cdot | x, y_{<t}))$ for the exploration/exploitation tradeoff. There is no mention as
  to why but one probable reason is that sampling inherently increases stochasticity.

&nbsp;

## Pipeline

### Original GRPO<sup>2, 3</sup>

We start with 3x from scratch models and OpenAI weights as our pretrained/base GPT-2 models (mostly for
testing the logic/exp than aiming for SOTA perfs...) based on [`llm_quest/gpt`](/llm_quest/gpt/):

- Policy model $\pi_{\theta}$:
    - SFT or Base: DeepSeek used a Base DeepSeekMath 7B model (for RLVR with R1 Zero), but here for RLHF and smaller
      models, it made more sense to do SFT first on the Alpaca dataset ([in instruction_tuning](/llm_quest/finetuning/instruction_tuning)).

    - $\pi_{\theta}$ is also used for $\pi_{\theta_{old}}$, not using a separate model<sup>2</sup> for sampling +
    policy ratio $\frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$


- Reference model $\pi_{ref}$:
    - a copy of $\pi_{\theta}$ which stays in evaluation mode, only used for KL divergence so
      that $\pi_{\theta}$ doesn't drift away too much or tricks $r_{\phi}$. 


- Reward model $r_{\phi}$:
    - We're first doing SFT (actually a copy of $\pi_{\theta}$) then pref training in `reward_model_training.py` on a preference (pref/rej pairs) dataset, with
    $r_{\phi}$'s output head changed to $W \in \mathbb{R}^{d_{out} \times 1}$ for a single scalar reward and aimed at
    minimizing the Bradley-Terry loss 
    $\mathcal{L}(\phi) = - \log \sigma(R_\phi(p, res_{\text{pref}}) - R_\phi(p, res_{\text{rej}}))$ *with $p$ for prompt.*   
    $r_{\phi}$ is used to compute the mini-rewards/per tokens of each generated trajectory from $\pi_{\theta}$.

### Visualization of the pipeline:

<img src="_mermaid_visual.png" alt="mermaid diagram" width="50%">

&nbsp;

*GRPO Algorithm's description from the DeepSeekMath paper below*, recopied for reference, imo, the `grpo_training_loop`
function in `grpo_engine.py`, pretty much reads by itself compared to the pseudocode, so I won't go into
unnecessary details.


> **Input:** initial policy model $\pi_{\theta_{init}}$; reward models $r_{\phi}$; task prompts $\mathcal{D}$; hyperparameters $\epsilon, \beta, \mu$
> 
> 1. policy model $\pi_{\theta} \leftarrow \pi_{\theta_{init}}$
> 2. **for** iteration = 1, ..., I **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp; reference model $\pi_{ref} \leftarrow \pi_{\theta}$
> 4. &nbsp;&nbsp;&nbsp;&nbsp; **for** step = 1, ..., M **do**
> 5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample a batch $\mathcal{D}_b$ from $\mathcal{D}$
> 6. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update the old policy model $\pi_{\theta_{old}} \leftarrow \pi_{\theta}$
> 7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample G outputs $\lbrace o_i \rbrace_{i=1}^G \sim \pi_{\theta_{old}}(\cdot | q)$ for each question $q \in \mathcal{D}_b$
> 8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute rewards $\lbrace r_i \rbrace_{i=1}^G$ for each sampled output $o_i$ by running $r_{\phi}$
> 9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute $\hat{A}_{i,t}$ for the t-th token of $o_i$ through group relative advantage estimation.
> 10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** GRPO iteration = 1, ..., $\mu$ **do**
> 11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update the policy model $\pi_{\theta}$ by maximizing the GRPO objective (Equation 21)
> 12. &nbsp;&nbsp;&nbsp;&nbsp; Update $r_{\phi}$ through continuous training using a replay mechanism.
> 
> **Output:** $\pi_{\theta}$


***1*:** *See [Experimental](#experimental)*

***2*:** *Concerning line 6 of GRPO, it may be a misinterpretation on my part (can be modified if wrong anyway).  
For efficiency purposes, since every new batch* $\pi_{\theta_{old}}$ = $\pi_{\theta}$ *we can use* $\pi_{\theta}$
*for old log probabilities and generating trajectories with carefully switching from `.eval()` +
`torch_inference_mode()` and `.train()` for gradient updates.*

***3***: *The last line (12) of the GRPO Algorithm, DeepSeek included it for completeness. This is concerning the
optional iterative RL alternative where the reward model is no longer frozen during RLHF, but continuously trained
(replay mechanism). I initially went PPO-style with pretraining & freezing a reward model (which is also described as
frozen in figure 4 of the paper), thus continuous* $r_{\phi}$ *update is not implemented atm.*

### Experimental (untested mix of GRPO iterative RL)

This is built on top of the assumptions ***1, 2*** from `grpo_training_loop` and the fact that for the iterative RL
variant, DeepSeek set the reference model as the policy model.  
In `grpo_training_loop_variant_experimental`, we take a risk by unanchoring $\pi_{ref}$ from
the outer loop and increasing its update frequency, keeping it much closer to $\pi_{\theta}$ by updating it every
batch/collection. ie $(\pi_{ref}, \pi_{\theta_{old}}) \leftarrow (\pi_{\theta}, \pi_{\theta})$

The positive point, we're left with a single model for handling $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$,
giving DPO-like vibes with only 2 models in total.  
The drawback is that $\pi_{ref}$ anchor role won't be as strong and bias will increase, since it's updated every batch with $\pi_{\theta_{old}}$.

2x models as:
- Policy model for $\pi_{\theta}$, $\pi_{\theta_{old}}$ and $\pi_{ref}$

    - $\pi_{\theta}$ will be used for KL divergence + policy ratio.
    - I'm only updating $\pi_{ref}$ per batch (ie, not epoch or gradient update if `num_grad_updates > 1`).

- Reward model $r_{\phi}$:
    - Same as original GRPO minus ***3*** mentioned above.

&nbsp;

## Results

The variance in the quality of the responses is high but as a baseline it was already the case with DPO or even SFT where
the model simply gives a wrong answer for lack of understanding or hallucinates.

One remarkable behavior I found, for such a small (355M params) and old architecture based model, was when it
learned by itself 2 traits:  
  - Humbleness and honesty.  

It really highlights the beauty of RL because the model never saw any data at all, in SFT or in the preference
dataset, that would influence its behavior in this direction.  
This is visible with SFT tests where the model always tries to follow the instruction no matter what.

Below are some cool examples showing this behavior, these samples were taken from the test set (unseen data), with a
`max_gen` = 35 tokens:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the chemical formula for ammonium nitrate?

Correct response:
>> The chemical formula for ammonium nitrate is NH4NO3.

Reference model response:
>> The chemical formula for ammonium nitrate is C2H4O3.

Policy model response:
>> I would like to suggest that you refer to the chemical formula of ammonium nitrate in order to learn more about the chemical formula.

(another run response)
>> I am not sure exactly what this chemical formula is, so I will not be able to give you an answer.

------------------------------------------------------------------------------------

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the formula for calculating work done?

Correct response:
>> The formula for calculating work done is work = force × distance.

Reference model response:
>> The formula for calculating work done is: "L=W+D/N." In other words, what is calculated as work done when L

Policy model response:
>> I have not been able to find a good answer to this question because my understanding of the problem is limited. This has not been a problem for
```

As small and old as it is, this GPT2-based model literally:
- takes a step back from the instruction given
- admits it's not really sure about the answer
- prefers to politely abstain rather than give a wrong answer or hallucinate something

Even kindly "suggested" that I refer to other sources myself to learn more about the topic.  

The way it replied on some of these prompts gave the impression it "thought" about it, before concluding it doesn't
know the answer. It was like it had undergone reasoning-focused training like RLVR beforehand.

A drawback of this run though, the model would tend to become overly hesitant (low self-confidence) way too often.


&nbsp;

## Additional details

### Outcome Supervision vs Process Supervision

I chose Outcome Supervision (1 reward per trajectory, unlike [Process Supervision](https://arxiv.org/abs/2312.08935))
concerning the reward calculations, for simplicity BUT computed in a way that is compatible with process supervision
because:

  My Reward model isn't an ORM but a PRM (Per-Token Reward Model) which gives me choices on how to represent the reward.
  I implemented different simple styles for testing: scores pooling, hidden states pooling and "last token only" in the `PrefRewardCalculator` class.

  Each has its pros and cons. I initially went with scores pooling, but it is less robust when there is high variance
  in sequence lengths.
    
  In the case of a later implementation of Process supervision, the PRM is compatible since I'm already
  retrieving all mini-rewards at each step for each trajectory, so I can reuse it with process supervision.


### Difficulties encountered

#### Gradient underflow

  Aggressive memory optimization, particularly casting models to bf16 and without using Automatic Mixed Precision (AMP),
  had major drawbacks. While it was acceptable for SFT, it caused issues during the reward model training and RLHF.  
  Specifically: small learning rates (e.g., 1e-5) resulted in gradient updates that were too small to be represented in
  bf16, subsequently rounded to zero, leading to stagnant training.  
  Switching to fp32 or increasing the learning rate (at risk) clearly mitigated the issue, (ie, showed the impact of precision on gradient representation during training.)

#### Reward hacking and model collapse

  Being too greedy chasing rewards, coupled with a generous KL div/relative entropy penalty, quickly led to punishment:
  The policy exploited the reward model up to a point of no return.  
  
  Below is a funny example, after the policy was exposed to excessive KL divergence (> 1, in my case)
  for too long, it tricked the RM by producing overly polite repetitive tokens, completely ignoring the prompt:


  ```
  ### Instruction:
  Rewrite the sentence using a simile.

  ### Input:
  The car is very fast.

  Correct response:
  >> The car is as fast as lightning.

  Reference model response:
  >> The car is very fast.

  Policy model response:
  >> I would suggest you use a simile to describe the car as it is. I would be very grateful if you would consider this suggestion. I would love to hear your thoughts and I would be more than happy to help
  ```

#### Rigorous prompts batching for generation

  We have to batch prompts for efficient generations in parallel, the problem is that the model will see padding tokens
  for shorter prompts during generation.

  **If we pad left**, so that it sees: `<EoS><EoS><Instructions...>+"### Response:" START GEN`
  - Pros: 
    - handy for generation as the last prediction retrieval will be guaranteed to be coming from a real token and not a
    padding one.  
    We can keep the typical `[:, -1, :]` slicing to retrieve the real last token's prediction.
  - Cons: 
    - Since the first tokens are padding, the model will inject positional information for them as the beginning of the
    sequence, and the real beginning of the sequence will get incorrect positional encoding. (*Maybe for large models
    this is less of a problem but for a small model like this one, it completely ruins the generation: spitting
    nonsense*) 

      *Possible solution*: keeping track of the real tokens' positions and using that for the positional encoding.

    - Classic attention mask with masking value as `-inf` will trigger an edge case where the first token (padding)
      will only attend to itself, resulting in an attention scores' vector of all `-inf` values, which will
      consequently trigger a softmax of all `NaN` values and ruin the training.  

      *Possible solution*: using `torch.finfo().min` as masking value (the loss mask will remove the padding tokens from
      the loss in any case).

  **If we keep pad right**: `<Instructions...>+"### Response:"<EoS><EoS> START GEN`
  Unless we're fine retrieving the prediction of the last EoS token, right padding has pros and cons too.

  - Pros:
    - Don't have to deal with positional encoding.
    - The model will never see padding tokens in its context for the first step/gen if we retrieve the last real token.

  - Cons:
    - Since the last token will not be real one but padding, we can't easily slice with `[:, -1, :]` to retrieve logits
      of the last real token for the first step.

      *Possible solution*: Via the collator, we can keep track of the position of the last real tokens for each prompt
      and pass it to the generation function and slice with real positions.


  In both cases, attention mask helps prevent the model from attending to padding tokens.  

  **3rd inefficient option**, is simply doing 1 prompt at a time and increasing `num_samples` (for better perfs).

  **Also helpful**:
  - Sorting the prompts/dynamic batching by length helps limit the amount of padding tokens per batch.

#### Hyperparameters tuning

  The RLHF GRPO training is the last step to tune, since I'm controlling the whole pipeline, improvements
  start from the very beginning with the quality of the SFT and RM training before even starting the RLHF phase. This is while being bound by the GPT-2 architecture and compute (all trainings remain under 16GB, hardware-friendly). Even small differences upstream matter later on.  
  So I didn't focus too much on hparams as the primary goal was to achieve a working from-scratch implementation of RLHF
  based on DeepSeek GRPO, thus there is ample room for improvement.
