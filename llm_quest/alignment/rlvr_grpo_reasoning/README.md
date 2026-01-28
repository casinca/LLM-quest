# TODO README

# Reasoning from scratch: RLVR with GRPO

This RLVR implementation is not based on a specific paper, but a mix of techniques used and depicted in the following resources:

- AI2 TULU 3: https://arxiv.org/abs/2411.15124
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- RLHF book by [@natolambert](https://github.com/natolambert)  *(also core author of TULU 3)*:
  https://rlhfbook.com/c/14-reasoning.html
- Structure/code re-used from https://github.com/casinca/LLM-quest/tree/master/llm_quest/alignment/rlhf_grpo
- Dataset: https://huggingface.co/datasets/openai/gsm8k

*Unlike RLHF with GRPO from scratch, where only `tiktoken` was a dependency, here I'm using HuggingFace's tokenizer, for
its efficient `decode` method.*


# Pipeline

Start with Pretrained GPT2 (OpenAI weights)

↓
SFT (amp) on Alpaca dataset


GSM8K Dataset
↓
reformat dataset with alpaca prompt style + DeepSeek R1 reasoning format
↓
SFT (amp) on X GSM8K samples for cold start (learning the reasoning format)


reward function (for now checking verifiable answer (outcome))

↓
RLVR with GRPO in bf16


# CURR STATE 

Pipeline is working, but terribly slow compared to RLHF because of the increased batch size, necessary for the reasoning
part to evolve/exploration
- Need to profile and see if it's just about that or if there's a leak/mistake somewhere






EDIT: problem is indeed batched generations bc of no kvcache likely

```Average Timings:
generate_batched_loop: 4.331731 seconds (over 10 calls)
batched_responses_collator: 0.000448 seconds (over 10 calls)
log_probs_per_token_old: 0.027819 seconds (over 10 calls)
log_probs_per_token_ref: 0.027777 seconds (over 10 calls)
reward_calculator: 0.023353 seconds (over 10 calls)
z_scores: 0.000150 seconds (over 10 calls)
grad_log_probs_per_token_policy: 0.035956 seconds (over 20 calls)
grad_loss_calc: 0.001996 seconds (over 20 calls)
grad_optimizer_step: 0.071828 seconds (over 20 calls)
grad_total_grad_step: 0.151435 seconds (over 20 calls)
```

## potential improvements:

- Make a custom gsm8k dataset sorted by difficulty (curriculum learning style?)
- test half of gsm8k with shortest Q+traces
- KVcache would improve the most
- better/refined (precise)/more efficient (token-wise) "instruction" prompt in alpaca_deepseek_format()
- making sure SFT is properly learned canary style testing 
- switch to process supervision (mixing VR and non VR evaluation) (maybe but later for experimenting + TODO mention in README)


# keep for readme

## Importance of Process supervision vs Outcome supervision

While it's just an SFT warmup to learn the reasoning format, and not RLVR yet, the example below could definitely
happen during RL, where the model gets the right answer but the reasoning completely wrong.  
With outcome supervision, the model would get a positive reward for the right answer and indirectly reward a wrong
reasoning trajectory.  
Hence the importance of process supervision and shaping proper rewards for the reasoning process.

```
### Instruction:
Below is a question concerning a math problem. Your role as an assistant is to reason step by step and provide the final answer to the problem. It is very important that you structure your response into 2 main sections: reasoning and answer. You must enclose your reasoning process in <think> </think> tags and final answer in <answer> </answer> tags. For example: <think> reasoning process here </think> <answer> answer here </answer>. Following the above instructions, try to solve the question:

### Input:
A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Correct response:
<think>It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</think> <answer>3</answer>

Model response:
<think>It takes 2 x 2 = <<2*2=2>>2 bolts of blue fiber
2 x 1.5 = <<2*1.5=2>>2 bolts of white fiber
So the total is 3 x 2 = <<3x2=3>>3 bolts</think> <answer>3</answer>
```








