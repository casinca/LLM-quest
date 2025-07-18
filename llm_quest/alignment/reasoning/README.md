# TODO

# Pipeline atm

Start with Pretrained GPT2 (OpenAI weights)
↓
convert to bfloat16 or not (testing both and see which is better)
↓
SFT on Alpaca dataset
↓
Test


GSM8K Dataset
↓
reformat dataset with alpaca prompt style + DeepSeek R1 reasoning format
↓
SFT on X GSM8K samples for cold start (learning the reasoning format)
↓
Test


Create a reward function (for now checking answer (outcome) to make sure it kinda works)
↓
RLVR with GRPO
↓
Test


# potential improvements:

- Make a custom gsm8k dataset sorted by difficulty (curriculum learning style?)
- test half of gsm8k with shortest Q+traces
- better/refined (precise)/more efficient (token-wise) "instruction" prompt in alpaca_deepseek_format()
- making sure SFT is properly learned canary style testing
- switch to process supervision

# checks:

- (reasoning rewards later) reflect on completion
- confirm the approach

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








