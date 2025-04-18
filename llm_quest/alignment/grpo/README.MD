# game plan GRPO from scratch on preference tuning

## PRE STEPS

we will take a GPT2 124M with reduced ctx length for everything, we need to fit 3x LLMs at peak on 1 GPU, just need to test the
impl is working as intended.

### policy model: we will instruct SFT the GPT base model the same way as we did for ch7, no changes.

### ref model: copy of the frozen policy model. no changes.

### reward model: 
- prepare preference dataset:
    - modify the preference DPO dataset to only parse for instruction/prefered/rejected. Need new formatting function.
    - ~~**Need to decide or find** what method of loss computation we use: last valid token or mean pooling?~~
        - Either case we'll need to modify the collate func to mask accordingly depending on the method

          TODO using @rasbt func for now, need to do our own with last token option argument

      - we want the dataloader to ouput tuple `input, preferred, rejected = batch` and concat or input+pref & input+rej?
      **mask in the collate accordingly!**
    
- ~~we will take the SFT policy model or base policy? (DeepseekMath is base for lower bias but doesn't really mater for the impl):~~
    - ~~change the output head FFN to project 1 scalar (reward score)~~
    - ~~change the loss function to BT model.~~
    - ~~training epochs?~~



## GRPO

- we will use the generate function to sample from the policy model 5 responses with custom topk or temperature (for
  diversity)
- need a pipeline to dynamically append our response together then pad + batch & attn mask to return for feeding the
  reward model properly
- we calculate the full trajectory reward of each sample (there is **no γ discounted future rewards decay**/time penalty
  ) + the KL div is **NOT** applied to intermediate rewards nor the full
 reward!  
- from each full reward, we then calculate their z-score, which will be our advantage functions for each trajectory.  
 **ie we do Outcome Supervision, not Process Supervision**, (no intermediate Advantage functions at each step.)
- we clip our `advantage function * policy ratio` and then we apply KL div (via unbiased estimator (Schulman, 2020)) to
  the clipped loss



# TODO

polishing:
  - add `num_batch` and `eval_iter` for reward_model_evaluation()
  - unify training and val losses in evaluate

# FOR THE README

Concerning the choice of a reward model: Base vs SFT model.  
Pros of Base is that you don't get that sweet SFT bias but since it's not SFT'd, the model will have to learn
instructions by itself, likely requiring more training that an SFT'd one.

DeepSeek used a Base DeepSeekMath 7B model.

# Note

- lets not get confused with DPO and preprocessing for the loss, here we're projecting to a single scalar, not a proba
  distrib over vocab size, thus there is not logits and label shifting.

- **We use mean pooling for the full reward calculation for each trajectory. Even if we use outcome supervision.**
  Because in case of later implementation of Process supervision, mean pooling will be naturally compatible since we
  retrieve all mini rewards at each steps for each trajectory. If we had used last token only method, it wouldn't be
  compatible.