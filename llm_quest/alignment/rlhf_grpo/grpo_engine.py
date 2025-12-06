import os

import torch
import torch.nn.functional as F

import config
from llm_quest.alignment.gspo.gspo_engine import gspo_loss, log_probs_per_seq
from llm_quest.generate import generate_batched_loop, generate_batched_loop_kv_cache, generate_loop
from llm_quest.utils import CheckpointEvaluator


def bt_loss(chosen_logits, rejected_logits, beta=1.0):
    """
    Compute the Bradley-Terry loss between chosen and rejected logits.
    This is the equivalent of the BCE loss, see dpo README equations 8 and 9.
    This is the loss used for the reward model training

    Args:
        chosen_logits (torch.Tensor): Logits of the chosen response. Shape: (batch_size)
        rejected_logits (torch.Tensor): Logits of the rejected response. Shape: (batch_size)
        beta (float): scaling factor

    Returns:
        torch.Tensor: The mean loss value over the batch. Shape: (1,)
    """
    logits = beta * (chosen_logits - rejected_logits)
    # we minimize in the sense that the more sigmoid approaches 1, the more log(sigmoid) approaches 0
    loss = -F.logsigmoid(logits)

    return loss.mean()


# NOTE: There are a lot of ways and research on what's the best way to represent the reward for a sequence.
# these are just some that I found common and intuitive in the field.
# - We could also mix and weight these together (like last+h.states) in case last is noisy.
# - We could take the last k states and weight or not (linearly, exp, ...) by positional order...
class PrefRewardCalculator:
    """
    Different ways to calculate a reward for a sequence, from the reward model:

    - scores_mean_pooling: project hidden states to a scalar and then mean pooling the scores/scalars
    - hidden_state_mean_pooling: mean pooling over the hidden states and then project to a scalar
    - last_token_score: retrieve the last real token's (EoS in our case) hidden state and project to a scalar
    """

    @staticmethod
    def scores_mean_pooling(rewards, reward_mask):
        """
        Args:
            rewards (torch.Tensor): scores/scalars from the model's head (b, s, 1)
            reward_mask (torch.Tensor): boolean mask of shape (b, s)

        Returns:
            scores (torch.Tensor): shape (b,)
        """
        return (rewards.squeeze(-1) * reward_mask).sum(dim=1) / reward_mask.sum(dim=1).clamp(min=1)

    @staticmethod
    def hidden_states_mean_pooling(hidden_states, reward_mask, model_head):
        """
        Args:
            hidden_states (torch.Tensor): shape (b, s, emb_dim)
            reward_mask (torch.Tensor): boolean mask of shape (b, s)
            model_head (nn.Linear): shape (emb_dim, 1)

        Returns:
            scores (torch.Tensor): shape (b,)
        """
        # shape: (b, s, emb_dim) * (b, s, 1) → (b, s, emb_dim)
        hidden_states = hidden_states * reward_mask.unsqueeze(-1)

        # mean pooling over the sequence length (b, s, emb_dim) → (b, emb_dim)
        mean_hidden_states = hidden_states.sum(dim=1) / reward_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)

        # shape: (b, emb_dim) →  (b, 1) → (b, )
        scores = model_head(mean_hidden_states)
        return scores.squeeze(-1)

    @staticmethod
    def last_token_score(hidden_states, attention_mask, model_head):
        """
        Args:
            hidden_states (torch.Tensor): shape (b, s, emb_dim)
            attention_mask (torch.Tensor): boolean mask of shape (b, s)
            model_head (nn.Linear): shape (emb_dim, 1)

        Returns:
            scores (torch.Tensor): shape (b,)
        """
        seq_lengths = attention_mask.sum(dim=-1)  # trick to retrieve the last real token's index per sequence
        seq_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)

        # shape: (b, s, emb_dim) → slicing (b, emb_dim) → (b, 1) → (b, )
        scores = model_head(hidden_states[seq_idx, seq_lengths - 1, :])  # -1 because 0-indexed
        return scores.squeeze(-1)


def reward_model_training_eval_loop_simple(
    train_loader,
    val_loader,
    reward_model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_num_batches=None,
    beta=1.0,
):
    """
    A simple training and evaluation loop for the reward model.

    Args:
        train_loader (DataLoader): DataLoader for training data batches.
        val_loader (DataLoader): DataLoader for validation data batches.
        reward_model (nn.Module): The reward model to train. It should output token-level rewards.
        optimizer (torch.optim.Optimizer): Optimizer for the reward model parameters.
        num_epoch (int): Total number of epochs to train for.
        eval_freq (int): Frequency (in training steps) at which to perform evaluation.
                        Also used as the training loss logging interval.
        eval_num_batches (int, optional): Number of batches to use for evaluation.
                                            If None, evaluate the whole validation set.
        beta (float, optional): Scaling factor for the Bradley-Terry loss. Defaults to 1.0.
    """
    step = 0
    interval_train_loss = 0.0
    interval_correct_pred = 0
    interval_train_count = 0

    tracking = {
        "train_losses": [],
        "val_losses": [],
        "train_acc": [],
        "val_acc": [],
    }
    chkp_eval = CheckpointEvaluator()
    reward_model.train()

    for epoch in range(1, num_epoch + 1):
        for batch in train_loader:  # batch is already on the correct device via the collate func
            step += 1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pref_rewards = reward_model(  # shape (b,)
                    batch["chosen"],
                    attn_mask=batch["chosen_attn_mask"],
                    reward_mask=batch["chosen_mask"],
                )
                rej_rewards = reward_model(
                    batch["rejected"],
                    attn_mask=batch["rejected_attn_mask"],
                    reward_mask=batch["rejected_mask"],
                )

                loss = bt_loss(pref_rewards, rej_rewards, beta=beta)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            interval_train_loss += loss.item()
            interval_correct_pred += (pref_rewards > rej_rewards).sum().item()  # count number of correct predictions
            interval_train_count += pref_rewards.shape[0]

            # --- eval ---
            if step % eval_freq == 0:
                # calculate metrics for the interval
                avg_interval_train_loss = interval_train_loss / eval_freq  # Avg loss per batch in interval
                avg_interval_train_acc = (
                    interval_correct_pred / interval_train_count if interval_train_count > 0 else 0.0
                )
                val_loss, val_acc = evaluate_reward_model(val_loader, reward_model, eval_num_batches, beta=beta)
                tracking["val_losses"].append(val_loss)
                tracking["val_acc"].append(val_acc)

                print(
                    f"Epoch: {epoch}, Step: {step} |",
                    f"T. loss: {avg_interval_train_loss:.5f}, V. loss: {val_loss:.5f} |",
                    f"T. acc: {avg_interval_train_acc * 100:.2f}%, V. acc: {val_acc * 100:.2f}%",
                )

                if chkp_eval.is_rm_accu_best(val_acc, val_loss):
                    save_path = os.path.join(
                        config.rlhf_grpo_checkpoint_dir,
                        f"best_rm_checkpoint_{step}_accu_{chkp_eval.max_accu_pref_rm:.3f}_loss_{val_loss:.3f}.pt",
                    )
                    torch.save(reward_model.state_dict(), save_path)

                # reset interval training metrics
                interval_train_loss = 0.0
                interval_correct_pred = 0
                interval_train_count = 0

    return tracking


def evaluate_reward_model(val_loader, reward_model, eval_num_batches=None, beta=1.0):
    """
    Evaluate the reward model on the validation set.

    Args:
        val_loader (DataLoader): DataLoader providing batches of chosen and rejected responses.
        reward_model (nn.Module): The reward model being evaluated.
        eval_num_batches (int, optional): Number of batches to evaluate. If None, evaluate the whole validation set.
        beta (float, optional): Scaling factor for the Bradley-Terry loss. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the validation set.
            - avg_acc (float): The average accuracy over the validation set
            (proportion of batches where chosen reward > rejected reward).
    """

    total_loss = 0.0
    correct = 0
    count = 0

    num_batches_to_eval = min(eval_num_batches, len(val_loader)) if eval_num_batches else len(val_loader)

    reward_model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(val_loader):

            if i >= num_batches_to_eval:
                break

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pref_rewards = reward_model(
                    batch["chosen"],
                    attn_mask=batch["chosen_attn_mask"],
                    reward_mask=batch["chosen_mask"],
                )
                rej_rewards = reward_model(
                    batch["rejected"],
                    attn_mask=batch["rejected_attn_mask"],
                    reward_mask=batch["rejected_mask"],
                )

                loss = bt_loss(pref_rewards, rej_rewards, beta=beta)
            total_loss += loss.item()

            # count the number of correct predictions
            correct += (pref_rewards > rej_rewards).sum().item()
            count += pref_rewards.shape[0]

        avg_loss = total_loss / num_batches_to_eval
        avg_acc = correct / count

    reward_model.train()
    return avg_loss, avg_acc


def rlhf_grpo_prompt_collator(prompts, pad_token_id=50256, custom_max_length=None, device=torch.device("cpu")):
    """
    Collate function to pad prompts of different lengths into a single tensor, preparing them for the policy model
    sample generations.

    Args:
        prompts (List[List[int]]): A list of lists of token IDs.
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        custom_max_length (int, optional): Maximum length of the padded sequences. If None, the maximum length
                is determined by the longest prompt in the batch.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cpu".

        Dict[str, torch.Tensor]: A dictionary containing:
            padded_prompts: Tensor of shape (batch_size, max_len) with padded prompt token IDs.
            prompt_masks: Boolean tensor of the same shape to keep track of padded tokens.
            last_real_pos: Tensor of shape (batch_size,) containing the position of the last real token in each prompt.
    """

    max_length = max(len(sample) for sample in prompts)

    if custom_max_length is not None:
        prompts = [prompt[:custom_max_length] for prompt in prompts]
        max_length = min(max_length, custom_max_length)

    padded_prompts = []
    prompt_masks = []
    last_real_pos = []

    for sample in prompts:
        prompt_len = len(sample)
        padding_needed = max_length - prompt_len

        padded_prompt = sample + [pad_token_id] * padding_needed
        prompt_mask = [True] * prompt_len + [False] * padding_needed

        last_real_pos.append(prompt_len - 1)  # 0-indexed
        padded_prompts.append(padded_prompt)
        prompt_masks.append(prompt_mask)

    padded_prompts = torch.tensor(padded_prompts)
    prompt_masks = torch.tensor(prompt_masks, dtype=torch.bool)
    last_real_pos = torch.tensor(last_real_pos, dtype=torch.long)

    return {
        "padded_prompts": padded_prompts.to(device),
        "prompt_masks": prompt_masks.to(device),
        "last_real_pos": last_real_pos.to(device),
    }


# NOTE: responses generated from `generate_batched_loop()` already have an EoS token at the end (unless
# truncated/max_gen) therefore nothing is added here, responses are already ready to be sliced for logprobs.
def batched_responses_collator(responses, prompt_masks, device="cuda", eos_ids=50256, pad_token_id=50256):
    """
    Prepare batched sampled responses for the reward model.

    Args:
        responses (torch.Tensor): shape (batch_size * num_samples, prompt_len + max_gen)
                                responses = prompt + policy's output as padded token IDs.
        prompt_masks (torch.Tensor): Boolean tensor of shape (batch_size * num_samples, prompt_len)
                                    This is used to differentiate between special tokens used in the prompt.
                                    ex: chat template (should be attended) and special ones used as padding (not
                                    attended).
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cuda".
        eos_ids (int | List[int], optional): Token ID(s) that signal end of text generation. Defaults to 50256.
        pad_token_id (int, optional): Token ID to use for padding. Defaults to 50256.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            padded_responses: shape (batch_size * num_samples, prompt_len + max_gen)
                responses = prompt + policy's output as padded token IDs.
            reward_masks: Boolean tensor of the same shape with masked: prompt + padding tokens*.
            attn_masks: Boolean tensor of the same shape with masked: padding tokens*

            *Except the first EoS/pad token in the response part.
    """
    len_prompt = prompt_masks.shape[1]

    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids_tensor = torch.tensor(eos_ids, device=device, dtype=torch.long)

    is_eos = torch.isin(responses, eos_ids_tensor)
    is_pad = responses == pad_token_id
    stop_mask = is_eos | is_pad  # True if either EoS or pad token
    stop_mask[:, :len_prompt] = False  # exclude EoS or pad tokens from the prompt part

    # trick to retrieve the first EoS/pad in the response part
    # ex: [False, False, True, True]= [0, 0, 1, 2], all previous tokens will be 0, first EoS/pad will be 1
    cumsum_mask = stop_mask.cumsum(dim=1)
    # True for: previous tokens(=0) + first EoS/pad(=1)
    attn_masks = cumsum_mask <= 1
    # among previous tokens, masking the padding tokens in the prompt part
    attn_masks[:, :len_prompt] = prompt_masks

    reward_masks = attn_masks.clone()
    reward_masks[:, :len_prompt] = False

    return {
        "padded_responses": responses.to(device),
        "reward_masks": reward_masks.to(device),
        "attn_masks": attn_masks.to(device),
    }


def z_scores(rewards, num_samples, dr_grpo=None):
    """
    Compute the z-scores of the masked rewards.
    This is the way DeepSeek calculate the advantages in Outcome Supervision.

    Args:
        rewards (torch.Tensor): Tensor of shape (B*,) containing the masked rewards.
        num_samples (int): Number of samples per group (simply used for reshaping per groups).
        dr_grpo (str, optional): if "dr_grpo", compute the advantages per the Dr. GRPO loss method.
                                (arg made as string and not a bool to match "loss_variant" string arg in training loop)

        *considering B as batch_size * num_samples.

    Returns:
        torch.Tensor: Tensor of shape (B*,) containing the z-scores.
    """
    # reshaping per groups, shape (batch_size, num_samples), to calculate the advantages.
    # (We don't want stats from different groups/prompts to affect one another.)
    rewards = rewards.view(-1, num_samples)  # batch size inferred (ie rewards.shape[0] // num_samples)

    # lazily add an extra 0 reward to the group to avoid the edge case of std=0 when all rewards are the same but != 0
    if config.use_phantom_reward:
        phantom_reward = torch.zeros(rewards.shape[0], 1, device=rewards.device)
        augmented_rewards = torch.cat([rewards, phantom_reward], dim=1)
    else:
        augmented_rewards = rewards

    group_mean = augmented_rewards.mean(dim=1, keepdim=True)

    if dr_grpo == "dr_grpo":
        z_scores = rewards - group_mean
    else:
        if not config.use_phantom_reward:
            assert num_samples > 1, "num_samples must be greater than 1 to get a relative comparison"
        group_std = augmented_rewards.std(dim=1, keepdim=True)
        z_scores = (rewards - group_mean) / (group_std + 1e-8)  # small epsilon to avoid the edge case div by zero

    return z_scores.view(-1)  # flattening back to (B,)


# Removed intermediate masking of logprobs:
# - see: /LLM-quest/llm_quest/alignment/rlhf_grpo/README.md#additional-details/intermediate-masking-of-logprobs
# - changed in commit: https://github.com/casinca/LLM-quest/commit/25bf58ebd8f21f1acb2fe3112bbd0005109bd7ae
def log_probs_per_token(logits, inputs):
    """
    Compute and retrieve the log probabilities assigned to each label in a sequence.
    This is similar to the compute_logprobs() method in the `DPOLoss` class.

    We are masking logprobs for prompt + padding tokens later with the loss_mask.

    Args:
        logits (torch.Tensor): Tensor of shape (B*, S*, vocab_size) containing the logits.
        inputs (torch.Tensor): Tensor of shape (B*, S*) containing the generated tokens from the policy.

        *considering B as batch_size * num_samples and S as prompt_len+max_gen.

    Returns:
        torch.Tensor: Tensor of shape (B, S-1)* containing the log probabilities.
    """
    logits = logits[:, :-1, :]
    labels = inputs[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)  # shape (b, s-1, v)

    # retrieving the log probs assigned to each label
    label_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze_(-1)

    return label_log_probs  # shape (b, s-1)


# Instead of first calculating the full distributions of log probabilities for each token and having a large (b, s-1, v)
# tensor to then filter log probs for the wanted/label tokens:
# We can filter directly the logits because logits are the same as unnormalized log probabilities (x_i = log(exp(x_i)))
# Therefore we just have to calculate and subtract the 2nd term via LogSumExp to get the log probs:
# log(P(x_i)) =  log(exp(x_i)) - log(sum(exp(x_j)) = x_i - log(sum(exp(x_j)))
#
# NOTE: As per Huggingface's TRL, this method has some problems with BF16 precision.
# which can be confirmed even here as some training batches will return loss 0.0000
# https://github.com/huggingface/trl/blob/0726977a3aaf893e594dc7c64aced8e90770f020/trl/trainer/utils.py#L1532C1-L1532C102
def log_probs_per_token_optimized(logits, inputs):
    """
    Compute and retrieve the log probabilities assigned to each label in a sequence.
    Shouldn't be used with BF16 or <=16 bits precision for now

    We are masking logprobs for prompt + padding tokens later with the loss_mask.

    Args:
        logits (torch.Tensor): Tensor of shape (B*, S*, vocab_size) containing the logits.
        inputs (torch.Tensor): Tensor of shape (B*, S*) containing the generated tokens from the policy.

        *considering B as batch_size * num_samples and S as prompt_len+max_gen.

    Returns:
        torch.Tensor: Tensor of shape (B, S-1)* containing the log probabilities.
    """
    logits = logits[:, :-1, :]
    labels = inputs[:, 1:]

    # select logits assigned to each label
    label_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # LogSumExp reduces logits to (b, s-1), we avoid creating a (b, s-1, v) tensor
    label_log_probs = label_logits - torch.logsumexp(logits, dim=-1)  # x_i - log(sum(exp(x_j)))

    return label_log_probs  # shape (b, s-1)


def kl_div_per_token(policy_logprobs, reference_logprobs):
    """
    Compute the KL divergence per token between the policy and reference log probabilities.
    Estimated with (Schulman, 2020) unbiased estimator, see:
    https://github.com/casinca/LLM-quest/tree/master/llm_quest/alignment/rlhf_grpo#grpo

    Args:
        policy_logprobs (torch.Tensor): Tensor of shape (B*, S*) containing the policy log probabilities.
        reference_logprobs (torch.Tensor): Tensor of shape (B*, S*) containing the reference log
        probabilities.

        *considering B as batch_size * num_samples and S as prompt_len+max_gen.

    Returns:
        torch.Tensor: Tensor of shape (B, S-1)* containing the KL divergence per token.
    """
    ratio = torch.exp(reference_logprobs - policy_logprobs)
    log_ratio = reference_logprobs - policy_logprobs

    return ratio - log_ratio - 1


def grpo_loss(
    policy_ratio,
    advantages,
    loss_mask,
    min_clip,
    max_clip,
    beta,
    kl_div,
    num_samples,
    max_gen=1,
    variant="grpo",
):
    """
    Compute original GRPO loss, DAPO, Dr. GRPO or GSPO variant for a batch.
    credit to @qgallouedec's TRL doc for enumerating the variants and the papers, which made it faster to implement.

    Args:
        policy_ratio (torch.Tensor): Tensor of shape (B, S-1) containing the policy ratio per token.
        advantages (torch.Tensor): Tensor of shape (B,) containing the advantages per response.
        loss_mask (torch.Tensor): Tensor of shape (B, S-1) containing the loss mask.
        min_clip (float): Minimum clip parameter ϵ for the clipped surrogate objective.
        max_clip (float): Maximum clip parameter ϵ for the clipped surrogate objective.
        beta (float): Beta for the KL divergence penalty term.
        kl_div (torch.Tensor): Tensor of shape (B, S-1) containing the KL divergence per token.
        num_samples (int): Number of samples per group (simply used for reshaping per groups).
        max_gen (int, optional): used for Length Bias in the Dr. GRPO loss (in p.7 of the Dr. GRPO paper)
                                Defaults to 1. (no effect)
        variant (str): Variant of the GRPO loss to compute, default is "grpo" alt: "dapo", "dr_grpo", "gspo"

    Returns:
        torch.Tensor: Tensor of shape (1,) containing the GRPO loss for a batch.
    """

    # depending on the policy ratio level, either we use GRPO token-level variants or the classic GSPO seq-level loss
    if variant == "gspo":
        grpo_loss_batch = gspo_loss(policy_ratio, advantages, min_clip, max_clip) # already masked for padding+prompt
    else:
        # (PyTorch will broadcast the advantages anyway, unsqueezing to emphasize advantages aren't per tokens)
        # ie, each trajectory gets a single advantage.
        advantages_broadcast = advantages.unsqueeze(-1)

        surr_obj_per_token = policy_ratio * advantages_broadcast
        clipped_surr_obj_per_token = torch.clip(policy_ratio, min=1 - min_clip, max=1 + max_clip) * advantages_broadcast

        grpo_loss_per_token = -(torch.min(surr_obj_per_token, clipped_surr_obj_per_token) - beta * kl_div)
        grpo_loss_per_token *= loss_mask  # final masking: prompt + padding tokens

        if variant == "grpo":
            # grpo loss per response
            grpo_loss_seq = grpo_loss_per_token.sum(-1) / loss_mask.sum(-1).clamp(min=1)

            # TODO (this part can be simplified to a single .mean() since groups are equal size)
            # if the vGRPO variant doesn't work, revert this
            # grpo loss per group/num_samples
            grpo_loss_group = grpo_loss_seq.view(-1, num_samples).mean(dim=1)
            # grpo loss for the batch
            grpo_loss_batch = grpo_loss_group.mean()

        # DAPO paper: https://arxiv.org/abs/2503.14476 equation 8 and "3.3 Rebalancing Act"
        # token-level averaging (longer seqs have more influence on the loss) instead of Sample-level: 1/n_G * sum(G_i)
        elif variant == "dapo":
            grpo_loss_batch = grpo_loss_per_token.sum() / loss_mask.sum().clamp(min=1)

        # Dr. GRPO paper: https://arxiv.org/abs/2503.20783 - Figure 1
        elif variant == "dr_grpo":
            grpo_loss_batch = grpo_loss_per_token.sum() / (grpo_loss_per_token.shape[0] * max_gen)

        else:
            raise ValueError(f"Unknown loss type: {variant}")

    return grpo_loss_batch


def sapo_loss(policy_ratio, advantages, loss_mask, temp_pos_tokens=1.0, temp_neg_tokens=1.05):
    """
    Compute the SAPO (Soft Adaptive Policy Optimization) loss from Qwen.
    https://arxiv.org/abs/2511.20347

    Args:
        policy_ratio (torch.Tensor): Tensor of shape (B, S-1) containing the policy ratio.
        advantages (torch.Tensor): Tensor of shape (B, 1) containing the advantages per sequence (broadcasted)
        loss_mask (torch.Tensor): Tensor of shape (B, S-1) containing the loss mask, this should be the reward mask.
        temp_pos_tokens (float, optional): Temperature (tau_pos in the paper) for positive tokens. Defaults to 1.0.
        temp_neg_tokens (float, optional): Temperature (tau_neg in the paper) for non-positive tokens. Defaults to 1.05.
                                            it's called "neg" but it's for <=0 advantages.

        Per the paper, fig.5 t_neg>t_pos, yields the best stability results.
    """
    temps = torch.where(advantages > 0, temp_pos_tokens, temp_neg_tokens)

    soft_gate = torch.sigmoid(temps * (policy_ratio - 1)) * 4 / temps
    sapo_loss_per_token = -soft_gate * advantages
    sapo_loss_per_token *= loss_mask

    sapo_loss_seq = sapo_loss_per_token.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    sapo_loss_batch = sapo_loss_seq.mean()

    return sapo_loss_batch


# NOTE: oldest GRPO version with single batch was removed in commit:
# https://github.com/casinca/LLM-quest/commit/da15e415aa91f70d34f4263b17df0e54e7c68f73


# At the time, the goal of this variant was to do training with 2 models only:
# - policy serving as π_θ, π_θ_old and π_ref
# - reward model serving as r_𝜑
#
# This doesn't seem to be that experimental after all since Qwen (for RLVR) doesn't even used kl div for their GSPO and
# Qwen3 model. Which basically resumes to having a single model + single reward function (or reward model for RLHF)
def grpo_training_loop_variant_experimental(
    train_loader,
    val_loader,
    policy_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    max_gen=35,
    min_clip_eps=0.2,
    max_clip_eps=0.2,
    beta=1.0,
    evaluation=True,
    eval_freq=None,
    eval_batches=None,
    eval_num_samples=1,
    kl_div_threshold=0.5,
    loss_variant="grpo",
    save_checkpoint=True,
):
    """
    GRPO training with just 2 models (1 policy + 1 reward model).
    This experimental version is based on 2 assumptions:
    - we can use a single policy for both π_θ and π_θ_old from the grpo_training_loop() func implementation.
    - we can use a single model for both π_ref and π_θ_old from a deprecated_grpo_training_loop_variant() I removed.
        see commit: https://github.com/casinca/LLM-quest/commit/21c477d247d2a7a2092f3eff951d41b02fc633b7

    We are left with a single model for π_θ, π_θ_old and π_ref.
    The drawback is that π_ref anchor role isn't as strong, since it's updated every sample/batch with π_θ_old.
    Thus a high beta would be recommended to compensate.

    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        val_loader (DataLoader): DataLoader providing batches of prompts for evaluation.
        policy_model (nn.Module): The language model being trained (policy π_θ) also used as π_θ_old and π_ref (KL div).
        reward_model (nn.Module): r_𝜑 pretrained to predict rewards for completions (frozen).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model for each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model (used for context length).
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        max_gen (int): Maximum number of tokens to generate for each response.
        min_clip_eps (float): Lower clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        max_clip_eps (float): Upper clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.
        evaluation (bool, optional): Whether to perform evaluation. Defaults to True.
        eval_freq (int, optional): Frequency (in training steps) at which to perform evaluation. Defaults to None.
        eval_batches (int, optional): Number of batches to evaluate on. If None, evaluates on the whole val_loader.
        eval_num_samples (int, optional): Number of responses to generate per prompt for evaluation. Defaults to 1.
        kl_div_threshold (float, optional): max KL divergence allowed for checkpoint saving. Defaults to 0.5.
        loss_variant (str, optional): Variant of the GRPO loss to compute, default is "grpo" alt: "dapo", "dr_grpo",
        "gspo".
        save_checkpoint (bool, optional): Whether to save the best checkpoint. Defaults to True.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()
    chkp_eval = CheckpointEvaluator(kl_div_threshold=kl_div_threshold, beta=beta)

    step = 0
    for epoch in range(1, num_epoch + 1):
        for batch in train_loader:
            step += 1
            policy_model.eval()  # for every new batch, π_θ, π_θ_old and π_ref are the same

            # --- Sampling responses ---
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            dup_prompts_masks = batch["prompt_masks"].repeat_interleave(num_samples, dim=0)
            last_real_pos = batch["last_real_pos"].repeat_interleave(num_samples, dim=0)

            responses = generate_batched_loop(
                input_tensor=dup_prompts,
                model=policy_model,
                attention_mask=dup_prompts_masks,
                max_gen=max_gen,
                context_length=policy_config["context_length"],
                top_k=20,
                temp=1,
                last_real=last_real_pos,
                device=device,
            )  # responses 2D shape: (batch_size * num_samples, max_prompt_len + max_gen), for simplicity: (B, S)

            collated_batch = batched_responses_collator(
                responses,
                prompt_masks=dup_prompts_masks,
                device=device,
            )

            # --- Retrieving logprobs & rewards ---
            with torch.inference_mode():
                loss_mask = collated_batch["reward_masks"][:, 1:]
                # we now use π_θ in inference mode for both π_ref and π_θ_old
                old_and_ref_logprobs = log_probs_per_token(  # shape: (B, S-1)
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )
                rewards = reward_model(  # shape: (B,)
                    collated_batch["padded_responses"],
                    attn_mask=collated_batch["attn_masks"],
                    reward_mask=collated_batch["reward_masks"],
                )

            advantages = z_scores(rewards, num_samples, dr_grpo=loss_variant)

            # --- Gradient updates loop ---
            policy_model.train()
            cum_grpo_loss = 0.0
            cum_kl_div = 0.0

            for grad_step in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )

                # π_ref = π_θ_old
                if loss_variant == "gspo":  # normalize policy ratio to the sequence level
                    pol_logprobs_per_seq = log_probs_per_seq(policy_logprobs, loss_mask)
                    old_logprobs_per_seq = log_probs_per_seq(old_and_ref_logprobs, loss_mask)
                    policy_ratio = torch.exp(pol_logprobs_per_seq - old_logprobs_per_seq)  # shape: (B,)
                else:  # token level policy ratio
                    policy_ratio = torch.exp(policy_logprobs - old_and_ref_logprobs)

                kl_div = kl_div_per_token(policy_logprobs, old_and_ref_logprobs)

                # loss, backprop, update
                grpo_loss_batch = grpo_loss(
                    policy_ratio=policy_ratio,
                    advantages=advantages,
                    loss_mask=loss_mask,
                    min_clip=min_clip_eps,
                    max_clip=max_clip_eps,
                    beta=beta,
                    kl_div=kl_div,
                    num_samples=num_samples,
                    variant=loss_variant,
                )

                optimizer.zero_grad()
                grpo_loss_batch.backward()
                optimizer.step()

                cum_grpo_loss += grpo_loss_batch.item()
                # using the training kl div as proxy for the evaluation here
                cum_kl_div += ((kl_div * loss_mask).sum() / loss_mask.sum()).item()

            avg_grpo_loss = cum_grpo_loss / num_grad_updates
            avg_kl_div = cum_kl_div / num_grad_updates

            # --- Evaluation ---
            if evaluation and eval_freq is not None and (step % eval_freq == 0):
                eval_metrics = GRPOEvaluator.evaluate(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    policy_model=policy_model,
                    reference_model=policy_model,  # won't be used for evaluation
                    evaluation_type="rlhf",
                    reward_model=reward_model,
                    policy_config=policy_config,
                    device=device,
                    max_gen=max_gen,
                    eval_num_samples=eval_num_samples,
                    eval_num_batches=eval_batches,
                )
                print(
                    f"Step {step} | "
                    f"Avg GRPO Loss: {avg_grpo_loss:.4f} | "
                    f"T. Rwd: {eval_metrics['train_reward']:.4f} | "
                    f"V. Rwd: {eval_metrics['val_reward']:.4f} | "
                    f"Avg KL Div (training): {avg_kl_div:.4f}"
                )

                # save new best checkpoint
                # (On policy KL div used as proxy)
                if save_checkpoint and chkp_eval.is_rlhf_grpo_best(avg_kl_div, eval_metrics["val_reward"]):
                    save_path = os.path.join(
                        config.rlhf_grpo_checkpoint_dir,
                        f"best_checkpoint_{step}_score_{chkp_eval.max_score_grpo:.3f}.pt",
                    )
                    torch.save(policy_model.state_dict(), save_path)


def rlhf_grpo_training_loop(
    train_loader,
    val_loader,
    policy_model,
    reference_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    max_gen=35,
    min_clip_eps=0.2,
    max_clip_eps=0.2,
    beta=1.0,
    evaluation=True,
    eval_freq=None,
    eval_batches=None,
    eval_num_samples=1,
    kl_div_threshold=0.5,
    loss_variant="grpo",
    save_checkpoint=True,
):
    """
    GRPO training loop.

    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (π_θ, also used for π_θ_old).
        reference_model (nn.Module): A copy of the policy model (as π_ref) used to compute:
                                    - KL divergence (D_KL(π_ref || π_θ)).
        reward_model (nn.Module): r_𝜑 pretrained to predict rewards for completions (frozen).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model for each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model (used for context length).
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        max_gen (int): Maximum number of tokens to generate for each response.
        min_clip_eps (float): Lower clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        max_clip_eps (float): Upper clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.
        evaluation (bool, optional): Whether to perform evaluation. Defaults to True.
        eval_freq (int, optional): Frequency (in training steps) at which to perform evaluation. Defaults to None.
        eval_batches (int, optional): Number of batches to evaluate on. If None, evaluates on the whole val_loader.
        eval_num_samples (int, optional): Number of responses to generate per prompt for evaluation. Defaults to 1.
        kl_div_threshold (float, optional): max KL divergence allowed for checkpoint saving. Defaults to 0.5.
        loss_variant (str, optional): Variant of the GRPO loss to compute, default is "grpo" alt: "dapo", "dr_grpo",
        "gspo".
        save_checkpoint (bool, optional): Whether to save the best checkpoint. Defaults to True.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()
    reference_model.eval()
    chkp_eval = CheckpointEvaluator(kl_div_threshold=kl_div_threshold, beta=beta)

    step = 0
    for epoch in range(1, num_epoch + 1):
        reference_model.load_state_dict(policy_model.state_dict())

        for batch in train_loader:
            step += 1
            policy_model.eval()  # for every new batch, π_θ and π_θ_old are the same
            # note: generate_loop() comes with torch.inference_mode() and to gpu device, no need to reapply here

            # --- Sampling responses ---
            # interleaving the prompts to generate multiple samples/responses in parallel
            # ex: batch size = 2, num_samples = 3 → [p1, p2] → [p1, p1, p1, p2, p2, p2]
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            dup_prompts_masks = batch["prompt_masks"].repeat_interleave(num_samples, dim=0)
            last_real_pos = batch["last_real_pos"].repeat_interleave(num_samples, dim=0)

            responses = generate_batched_loop_kv_cache(
                input_tensor=dup_prompts,
                model=policy_model,
                attention_mask=dup_prompts_masks,
                max_gen=max_gen,
                context_length=policy_config["context_length"],
                top_k=20,
                top_p=None,
                min_p=None,
                temp=1.0,
                last_real=last_real_pos,
                device=device,
                rope_model=False,
            )  # responses 2D shape: (batch_size * num_samples, max_prompt_len + max_gen), for simplicity: (B, S)

            collated_batch = batched_responses_collator(
                responses,
                prompt_masks=dup_prompts_masks,
                device=device,
            )

            # --- Retrieving logprobs & rewards ---
            with torch.inference_mode():
                loss_mask = collated_batch["reward_masks"][:, 1:]  # matching token positions corresponding to logprobs

                old_logprobs = log_probs_per_token(  # shape: (B, S-1)
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )
                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )

                rewards = reward_model(  # shape: (B,)
                    collated_batch["padded_responses"],
                    attn_mask=collated_batch["attn_masks"],
                    reward_mask=collated_batch["reward_masks"],
                )

            advantages = z_scores(
                rewards, num_samples, dr_grpo=loss_variant
            )  # grouping and computing zscores (outside the inference scope)

            # --- Gradient updates loop ---
            policy_model.train()
            cum_grpo_loss = 0.0

            for grad_step in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )

                if loss_variant == "gspo":  # normalize policy ratio to the sequence level
                    pol_logprobs_per_seq = log_probs_per_seq(policy_logprobs, loss_mask)
                    old_logprobs_per_seq = log_probs_per_seq(old_logprobs, loss_mask)
                    policy_ratio = torch.exp(pol_logprobs_per_seq - old_logprobs_per_seq)  # shape: (B,)
                else:  # token level policy ratio
                    policy_ratio = torch.exp(policy_logprobs - old_logprobs)

                kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)  # (will be masked in the loss calc)

                # loss, backprop, update
                grpo_loss_batch = grpo_loss(
                    policy_ratio=policy_ratio,
                    advantages=advantages,
                    loss_mask=loss_mask,
                    min_clip=min_clip_eps,
                    max_clip=max_clip_eps,
                    beta=beta,
                    kl_div=kl_div,
                    num_samples=num_samples,
                    variant=loss_variant,
                )

                optimizer.zero_grad()
                grpo_loss_batch.backward()
                optimizer.step()

                cum_grpo_loss += grpo_loss_batch.item()

            avg_grpo_loss = cum_grpo_loss / num_grad_updates

            # --- Evaluation ---
            if evaluation and eval_freq is not None and (step % eval_freq == 0):
                eval_metrics = GRPOEvaluator.evaluate(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    evaluation_type="rlhf",
                    reward_model=reward_model,
                    policy_config=policy_config,
                    device=device,
                    max_gen=max_gen,
                    eval_num_samples=eval_num_samples,
                    eval_num_batches=eval_batches,
                )
                print(
                    f"Step {step} | "
                    f"Avg GRPO Loss: {avg_grpo_loss:.4f} | "
                    f"T. Rwd: {eval_metrics['train_reward']:.4f}, T. KL Div: {eval_metrics['train_kl_div']:.4f} | "
                    f"V. Rwd: {eval_metrics['val_reward']:.4f}, V. KL Div: {eval_metrics['val_kl_div']:.4f}"
                )

                # save new best checkpoint
                if save_checkpoint and chkp_eval.is_rlhf_grpo_best(
                    eval_metrics["val_kl_div"], eval_metrics["val_reward"]
                ):
                    save_path = os.path.join(
                        config.rlhf_grpo_checkpoint_dir,
                        f"best_checkpoint_{step}_score_{chkp_eval.max_score_grpo:.3f}.pt",
                    )
                    torch.save(policy_model.state_dict(), save_path)


# NOTE: GPT2 RLHF written tests on preference tuning was done with the old `generate_batched_loop()` function that has
# been replaced here with `generate_batched_loop_kv_cache()` function.
# If want to re-use the old `generate_batched_loop()` function, we need to change it in:
# - _compute_grpo_metrics()
# - rlhf_grpo_training_loop()
class GRPOEvaluator:
    """
    Evaluator class for GRPO that works for both RLHF and RLVR.
    Computes the average reward and KL divergence of the policy model on both training and validation datasets.
    """

    @staticmethod
    def _compute_grpo_metrics(
        loader,
        policy_model,
        reference_model,
        policy_config,
        device,
        max_gen,
        eval_num_samples,
        eval_num_batches,
        evaluation_type="rlhf",
        reward_model=None,
        reward_calculator=None,
        rope_model=False,
        eos_ids=50256,
        pad_id=50256,
        sampling_params=None,
    ):
        total_reward = 0.0
        total_kl_div = 0.0
        num_batches_to_eval = min(eval_num_batches, len(loader)) if eval_num_batches else len(loader)

        for i, batch in enumerate(loader):
            if i >= num_batches_to_eval:
                break

            # --- Sampling responses ---
            dup_prompts = batch["padded_prompts"].repeat_interleave(eval_num_samples, dim=0)
            dup_prompts_masks = batch["prompt_masks"].repeat_interleave(eval_num_samples, dim=0)
            last_real_pos = batch["last_real_pos"].repeat_interleave(eval_num_samples, dim=0)
            responses = generate_batched_loop_kv_cache(
                input_tensor=dup_prompts,
                model=policy_model,
                attention_mask=dup_prompts_masks,
                max_gen=max_gen,
                context_length=policy_config["context_length"],
                last_real=last_real_pos,
                device=device,
                rope_model=rope_model,
                eos_ids=eos_ids,
                pad_id=pad_id,
                **(sampling_params if sampling_params is not None else {}),
            )

            collated_batch = batched_responses_collator(
                responses,
                prompt_masks=dup_prompts_masks,
                device=device,
            )

            loss_mask = collated_batch["reward_masks"][:, 1:]

            # --- Get logprobs from policy and reference models ---
            policy_logprobs = log_probs_per_token(
                logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                inputs=collated_batch["padded_responses"],
            )
            reference_logprobs = log_probs_per_token(
                logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                inputs=collated_batch["padded_responses"],
            )

            # --- Retrieving rewards ---
            if evaluation_type == "rlhf":
                rewards = reward_model(
                    collated_batch["padded_responses"],
                    attn_mask=collated_batch["attn_masks"],
                    reward_mask=collated_batch["reward_masks"],
                )

            elif evaluation_type == "rlvr":
                # duping answers to match the number of samples
                correct_answers = [ans for ans in batch["answers"] for _ in range(eval_num_samples)]
                rewards = reward_calculator(
                    collated_batch["padded_responses"],
                    correct_answers,
                )

            mean_batch_rewards = rewards.mean()

            # --- KL Divergence ---
            # here masking KL div since we are also printing it for the correct tokens.
            kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)
            masked_kl_div = kl_div * loss_mask
            mean_batch_kl_div = (masked_kl_div.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)).mean()

            total_reward += mean_batch_rewards.item()
            total_kl_div += mean_batch_kl_div.item()

        avg_reward = total_reward / num_batches_to_eval
        avg_kl_div = total_kl_div / num_batches_to_eval

        return {"reward": avg_reward, "kl_div": avg_kl_div}

    @staticmethod
    def evaluate(
        train_loader,
        val_loader,
        policy_model,
        reference_model,
        policy_config,
        device,
        max_gen,
        evaluation_type="rlhf",
        reward_model=None,
        reward_calculator=None,
        eval_num_samples=1,
        eval_num_batches=None,
        rope_model=False,
        eos_ids=50256,
        pad_id=50256,
        sampling_params=None,
    ):
        """
        Evaluates the performance of the policy model on both training and validation datasets.
        Args:
            train_loader (DataLoader): DataLoader for the training prompts.
            val_loader (DataLoader): DataLoader for the validation prompts.
            policy_model (nn.Module): The policy model to evaluate.
            reference_model (nn.Module): The reference model for KL divergence calculation.
            evaluation_type (str): The type of evaluation to perform ("rlhf" or "rlvr").
            reward_model (nn.Module, optional): The reward model to score generated responses (for RLHF).
            reward_calculator (Callable, optional): The reward calculator to score generated responses (for RLVR).
            policy_config (dict): Configuration dictionary for the policy model (used for context length).
            device (str): The device to run evaluation on.
            max_gen (int): Maximum number of tokens to generate for each response.
            rope_model (bool, optional): Whether to use a model which uses RoPE (backward compatibility with GPT2)
            eos_ids (int | List[int], optional): Token ids to use for the end of sequence.
            pad_id (int, optional): Token id to use for padding.
            eval_num_samples (int): Number of responses to generate per prompt. Defaults to 1.
            eval_num_batches (int, optional): Number of batches to evaluate on. If None, evaluates on the whole val_loader.
            sampling_params (dict, optional): Dictionary containing sampling parameters (top_k, top_p, min_p, temp).
        Returns:
            dict[str, float]: A dictionary containing evaluation metrics: average reward and KL divergence.
        """
        # checks for the evaluation type
        if evaluation_type == "rlhf":
            if reward_model is None:
                raise ValueError("reward_model is required for RLHF evaluation")
            reward_model.eval()
        elif evaluation_type == "rlvr":
            if reward_calculator is None:
                raise ValueError("reward_calculator is required for RLVR evaluation")
        else:
            raise ValueError(f"Invalid evaluation type: {evaluation_type}")

        policy_model.eval()
        reference_model.eval()

        with torch.inference_mode():
            train_metrics = GRPOEvaluator._compute_grpo_metrics(
                train_loader,
                policy_model,
                reference_model,
                policy_config,
                device,
                max_gen,
                eval_num_samples,
                eval_num_batches,
                evaluation_type=evaluation_type,
                reward_model=reward_model,
                reward_calculator=reward_calculator,
                rope_model=rope_model,
                eos_ids=eos_ids,
                pad_id=pad_id,
                sampling_params=sampling_params,
            )
            val_metrics = GRPOEvaluator._compute_grpo_metrics(
                val_loader,
                policy_model,
                reference_model,
                policy_config,
                device,
                max_gen,
                eval_num_samples,
                eval_num_batches,
                evaluation_type=evaluation_type,
                reward_model=reward_model,
                reward_calculator=reward_calculator,
                rope_model=rope_model,
                eos_ids=eos_ids,
                pad_id=pad_id,
                sampling_params=sampling_params,
            )

        policy_model.train()

        return {
            "train_reward": train_metrics["reward"],
            "train_kl_div": train_metrics["kl_div"],
            "val_reward": val_metrics["reward"],
            "val_kl_div": val_metrics["kl_div"],
        }
