# Reinforcement Pretraining (RPT) from scratch

Reinforcement Pretraining is a clever idea from Microsoft researchers: https://arxiv.org/abs/2506.08007. As the name
suggests, mixing RL with a pretraining objective. They reframe next token prediction (NTP) as a reasoning task, giving
birth to Next Token Reasoning (NTR?).

BUT the beauty of RPT is that it isn't locked to NTP, on the contrary, it extends to Multi Token prediction (MTP)
elegantly/naturally. All of this without modifying the model architecture (compared to how MTP is done in DeepSeek V3,
with extra attached modules or the [original MTP paper](https://arxiv.org/abs/2404.19737) from Meta researchers).

## Details

The 2-checks reward function they elaborated (see *3.2 Pre-Training with Reinforcement Learning in p.4* or the
`PrefixMatchingReward` class in `rpt_engine.py`) allows the model to be evaluated against, not just a single label (like
NTP), but a defined amount of labels as seen in the `RPTStructuredDataset` in `llm_quest/dataset.py` with the
`labels_length` hparam/argument.  
For example, for a single sample, if the context is: `"The capital of Fra"` and labels are `"nce is Paris and it's"`,
the model will be rewarded for predicting `"nce is Paris"` in 1 step.

Note: At the time of writing, from the current state of the paper, the "Pretraining" in RPT doesn't have the same
meaning as classic pretraining. The starting point isn't an untrained model but an already
pretrained+SFT+aligned/reasoning model.  
The authors do acknowledge this in *p.9* and mention it's an actual topic of research: doing RPT starting from an
earlier phase of the pipeline/stack.

## RPT benefits

Per the authors' results (*4. Experiments on p.5*), RPT could bridge to/unlock a new pipeline, after reasoning, for
difficult tasks:  
The RPT'ed model is not just better at high entropy NTP, but NTP in general (ref *table 1* from the paper) than a
classic reasoning model, thus having a model with better autoregressive foundations. It's not crazy to hypothesize that
it could be better at SFT, better at being aligned, better at reasoning again. Creating a virtuous enhanced cycle.  
This is what the researchers tested in *4.3 Reinforcement Fine-Tuning with RPT* showing better results for RLVR (not
only post training but also pre RLVR training. So a better starting and ending checkpoint for RLVR with the RPT model).


## RPT: structured datasets vs continuous corpus of text

RPT was done on structured datasets (Omni-Math, Tabular type etc...), ie with independent samples. Doing entropy
filtering on these samples is straightforward because each sample has a truly distinct and defined beginning and end.

Doing entropy filtering for RPT on a continuous corpus of text (like classic pretraining) is a bit more tricky because
as we slide the window of context, the first token of each sample is considered a beginning, but in reality, it'll
likely be a continuation of the previous sample.  
This falsely leads the first tokens' predictions of each sample into high entropy predictions, because they are
considered as having no prior context and thus harder to predict.  
We could add a $x$ number of tokens, as prefix, to give some context to the first tokens of each sample, but haven't
really digged into it.

## Test: TODO if possible

There is a reason the authors did RPT starting from a decent Deepseek-R1-Distill-Qwen-14B model. Despite the RPT
training working from a coding perspective, I'm still blind on results, the from scratch 300M reasoning GPT-2 is
simply not good enough in its current state for testing if learning is really happening or catching latent problems.

