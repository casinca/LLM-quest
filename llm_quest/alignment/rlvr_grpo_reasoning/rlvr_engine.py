import os

import torch

import config
from llm_quest.alignment.gspo.gspo_engine import log_probs_per_seq
from llm_quest.alignment.rlhf_grpo.grpo_engine import (
    GRPOEvaluator,
    batched_responses_collator,
    grpo_loss,
    kl_div_per_token,
    log_probs_per_token,
    off_policy_seq_mask,
    z_scores,
)
from llm_quest.generate import generate_batched_loop_kv_cache
from llm_quest.utils import CheckpointEvaluator, ResponseExtractor


class VerifiableRewardCalculator:
    """
    Reward calculator for a batch of responses, based on simple heuristics.

    Args:
        tokenizer (Tokenizer): The tokenizer to decode the responses (needs a batch_decode method).
        good_answer_reward (float): The reward value for a correct answer.
        wrong_answer_reward (float): The penalty for an incorrect answer (should be ≤ 0).
        unfinished_answer_reward (float): The penalty for an unfinished answer (should be ≤ 0).
        reasoning_weight (float): A coeff to weight the reasoning reward vs. the answer.
        pad_token_id (int): The token id to use for padding.
        dtype (torch.dtype): The dtype of the returned rewards.
    """

    def __init__(
        self,
        tokenizer,
        good_answer_reward=10.0,
        wrong_answer_reward=0.0,
        unfinished_answer_reward=-1.0,
        reasoning_weight=0.0,
        dtype=torch.bfloat16,
        pad_token_id=50256,  # placeholder for now
    ):
        assert wrong_answer_reward <= 0, "wrong_answer_reward should be ≤ 0"
        assert unfinished_answer_reward <= 0, "unfinished_answer_reward should be ≤ 0"

        self.tokenizer = tokenizer
        self.good_answer_reward = good_answer_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.unfinished_answer_reward = unfinished_answer_reward
        self.reasoning_weight = reasoning_weight  # placeholder for now in case I want to do something fancy
        self.dtype = dtype

    def _calc_answer_reward(self, response_strings, correct_answers):
        """
        Calculate the rewards based on the model's answers, for a batch.

        Args:
            response_strings (list[str]): The decoded model's responses.
            correct_answers (list[str]): The correct/ground truth answers.

        Returns:
            list[float]: The rewards for a batch.

        """
        rewards_list = []

        for response_string, correct_answer in zip(response_strings, correct_answers):

            raw_model_answer = ResponseExtractor.get_answer(response_string)
            sanitized_model_answer = ResponseExtractor.sanitize_answer(raw_model_answer)
            sanitized_correct_answer = ResponseExtractor.sanitize_answer(correct_answer)

            if sanitized_model_answer is None:
                rewards_list.append(self.unfinished_answer_reward)
                continue

            try:
                if float(sanitized_model_answer) == float(sanitized_correct_answer):
                    rewards_list.append(self.good_answer_reward)
                else:
                    rewards_list.append(self.wrong_answer_reward)
            except ValueError:
                print(
                    f"Failed to convert answer to float: model_answer='{sanitized_model_answer}', "
                    f"correct_answer='{sanitized_correct_answer}'"
                )

        return rewards_list

    def _calc_reasoning_reward(self):
        pass
        # For now, following DeepSeek and AI2 TULU's impl, returning the answer only

    def __call__(self, model_responses, correct_answers):
        """
        Main orchestrator for the rewards' calculation.

        Args:
            model_responses (torch.Tensor): The model's responses, shape (B, S)
            correct_answers (list[str]): The correct/ground truth answers.

        Returns:
            torch.Tensor: The total rewards for a batch of responses, shape (B,)
            (total_rewards = answer_rewards atm)

        """
        decoded_responses = self.tokenizer.batch_decode(model_responses, skip_special_tokens=True)
        answer_rewards = self._calc_answer_reward(decoded_responses, correct_answers)

        return torch.tensor(answer_rewards, dtype=self.dtype, device=model_responses.device)


def rlvr_grpo_prompt_collator(batch, pad_token_id=50256, custom_max_length=None, device=torch.device("cpu")):
    """
    Collate function to pad prompts of different lengths into a single tensor, preparing them for the policy model
    sample generations. It also passes through the answers kept as strings.
    This is a slight variation from the original rlhf_grpo_prompt_collator() to return the answers.

    Args:
        batch (List[Dict[str, any]]): A list of samples from ReasoningDataset, each a dict with "prompt" and "answer".
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        custom_max_length (int, optional): Maximum length of the padded sequences. If None, the maximum length
                is determined by the longest prompt in the batch.
        device (torch.device or str, optional): Device where the resulting tensors will be placed. Defaults to "cpu".

    Returns:
        Dict[str, torch.Tensor or list]: A dictionary containing:
            padded_prompts: Tensor of shape (batch_size, max_len) with padded prompt token IDs.
            prompt_masks: Boolean tensor of the same shape to keep track of padded tokens.
            last_real_pos: Tensor of shape (batch_size,) containing the position of the last real token in each prompt.
            answers: List of answer strings.
    """
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] if "answer" in item else item["labels"] for item in batch]

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
        "answers": answers,
    }


def rlvr_grpo_training_loop(
    train_loader,
    val_loader,
    policy_model,
    reference_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    reward_calculator,
    max_gen=70,
    eos_ids=50256,
    pad_id=50256,
    min_clip_eps=0.2,
    max_clip_eps=0.2,
    beta=1.0,
    unbiased_kl_estimate=False,
    evaluation=True,
    eval_freq=None,
    eval_batches=None,
    eval_num_samples=1,
    kl_div_threshold=0.5,
    min_reward_threshold=0.35,
    loss_variant="grpo",
    save_checkpoint=True,
    rope_model=False,
    lr_scheduler=None,
    sampling_params=None,
    off_policy_sequence_masking=False,
):
    """
    Reinforcement Learning with Verifiable Rewards (RLVR) training loop with GRPO, derived from
    rlhf_grpo_training_loop().

    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (π_θ, also used for π_θ_old).
        reference_model (nn.Module): A copy of the policy model (as π_ref) used to compute:
                                    - KL divergence (D_KL(π_ref || π_θ)).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model for each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model (used for context length).
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        reward_calculator (Callable): A callable object that calculates the rewards for a batch of responses.
        max_gen (int): Maximum number of tokens to generate for each response.
        eos_ids (int | List[int]): Token ids to use for the end of sequence.
        pad_id (int): Token id to use for padding.
        min_clip_eps (float): Lower clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        max_clip_eps (float): Upper clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.
        unbiased_kl_estimate (bool, optional): Whether to use the Unbiased KL Estimate from the DeepSeek V3.2 paper.
                    Defaults to False.
        evaluation (bool, optional): Whether to perform evaluation. Defaults to True.
        eval_freq (int, optional): Frequency (in training steps) at which to perform evaluation. Defaults to None.
        eval_batches (int, optional): Number of batches to evaluate on. If None, evaluates on the whole val_loader.
        eval_num_samples (int, optional): Number of responses to generate per prompt for evaluation. Defaults to 1.
        kl_div_threshold (float, optional): max KL divergence allowed for checkpoint saving. Defaults to 0.5.
        min_reward_threshold (float, optional): minimum reward threshold for checkpoint saving. Defaults to 0.35.
        loss_variant (str, optional): Variant of the GRPO loss to compute, default is "grpo" alt: "dapo", "dr_grpo",
        "gspo".
        save_checkpoint (bool, optional): Whether to save the best checkpoint. Defaults to True.
        rope_model (bool, optional): Whether to use a model which uses RoPE (backward compatibility with GPT2)
        lr_scheduler (LearningRateScheduler, optional): Learning rate scheduler. Defaults to None.
        sampling_params (dict, optional): Dictionary containing sampling parameters (top_k, top_p, min_p, temp).
        off_policy_sequence_masking (bool, optional): Whether to use off-policy sequence masking. Defaults to False.
    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reference_model.eval()

    chkp_eval = CheckpointEvaluator(
        kl_div_threshold=kl_div_threshold, min_reward_threshold=min_reward_threshold, beta=beta
    )

    step = 0
    for epoch in range(1, num_epoch + 1):
        reference_model.load_state_dict(policy_model.state_dict())

        for batch in train_loader:
            policy_model.eval()  # for every new batch, π_θ and π_θ_old are the same
            # note: generate_loop() comes with torch.inference_mode() and to gpu device, no need to reapply here

            # --- Sampling responses ---
            # interleaving the prompts to generate multiple samples/responses in parallel
            # ex: batch size = 2, num_samples = 3 → [p1, p2] → [p1, p1, p1, p2, p2, p2]
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            dup_prompts_masks = batch["prompt_masks"].repeat_interleave(num_samples, dim=0)
            last_real_pos = batch["last_real_pos"].repeat_interleave(num_samples, dim=0)
            correct_answers = [ans for ans in batch["answers"] for _ in range(num_samples)]

            responses = generate_batched_loop_kv_cache(
                input_tensor=dup_prompts,
                model=policy_model,
                attention_mask=dup_prompts_masks,
                max_gen=max_gen,
                context_length=policy_config["context_length"],
                last_real=last_real_pos,
                rope_model=rope_model,
                device=device,
                eos_ids=eos_ids,
                pad_id=pad_id,
                **(sampling_params if sampling_params is not None else {}),
            )  # responses 2D shape: (batch_size * num_samples, max_prompt_len + max_gen), for simplicity: (B, S)

            collated_batch = batched_responses_collator(
                responses,
                prompt_masks=dup_prompts_masks,
                device=device,
            )

            # --- Retrieving logprobs & rewards ---
            with torch.inference_mode():
                loss_mask = collated_batch["reward_masks"][:, 1:]

                old_logprobs = log_probs_per_token(  # shape: (B, S-1)
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )

                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                )

                rewards = reward_calculator(  # shape: (B,)
                    collated_batch["padded_responses"],
                    correct_answers,
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

                # KL divergence will be masked in the loss calc
                if unbiased_kl_estimate:
                    kl_div = kl_div_per_token(policy_logprobs, reference_logprobs, policy_ratio=policy_ratio)
                else:
                    kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)

                if off_policy_sequence_masking:
                    # KL div (as π_θ_old/π_θ) approximated with K1 estimator
                    k1_estimator_per_token = old_logprobs - policy_logprobs.detach()
                    off_policy_mask = off_policy_seq_mask(k1_estimator_per_token, advantages, loss_mask, delta=0.1)
                    loss_mask &= off_policy_mask  # inject in the loss mask directly

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
                if lr_scheduler is not None:
                    lr_scheduler.step(step)
                grpo_loss_batch.backward()
                optimizer.step()
                step += 1

                cum_grpo_loss += grpo_loss_batch.item()

            avg_grpo_loss = cum_grpo_loss / num_grad_updates

            # --- Evaluation ---
            if evaluation and eval_freq is not None and (step == 1 or step % eval_freq == 0):
                eval_metrics = GRPOEvaluator.evaluate(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    evaluation_type="rlvr",
                    reward_calculator=reward_calculator,
                    policy_config=policy_config,
                    device=device,
                    max_gen=max_gen,
                    eval_num_samples=eval_num_samples,
                    eval_num_batches=eval_batches,
                    rope_model=rope_model,
                    eos_ids=eos_ids,
                    pad_id=pad_id,
                    sampling_params=sampling_params,
                )

                print(
                    f"Step {step} | "
                    f"Avg GRPO Loss: {avg_grpo_loss:.4f} | lr: {(lr_scheduler.current_lr if lr_scheduler is not None else optimizer.param_groups[0]['lr']):.1e} | "
                    f"T. Rwd: {eval_metrics['train_reward']:.4f}, T. KL Div: {eval_metrics['train_kl_div']:.4f} | "
                    f"V. Rwd: {eval_metrics['val_reward']:.4f}, V. KL Div: {eval_metrics['val_kl_div']:.4f}"
                )

                # save new best checkpoint
                if save_checkpoint and chkp_eval.is_rlvr_grpo_best(
                    eval_metrics["val_kl_div"], eval_metrics["val_reward"]
                ):
                    save_path = os.path.join(
                        config.rlvr_grpo_checkpoint_dir,
                        f"best_checkpoint_{step}_score_{chkp_eval.max_score_grpo:.3f}.pt",
                    )
                    torch.save(policy_model.state_dict(), save_path)


# quick test
if __name__ == "__main__":
    import transformers

    string_responses = [
        (
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "<think>Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.</think>\n <answer>72 </answer><|endoftext|>",
        ),
        (
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "<think>Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.</think>\n <answer>-72</answer><|endoftext|>",
        ),
    ]
    correct_answers = ["72.0", "-72 "]

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    reward_calculator = VerifiableRewardCalculator(tokenizer=tokenizer)

    # Encode all responses in batch
    encoded_responses = [tokenizer.encode(response[1]) for response in string_responses]
    print(f"Encoded responses: {encoded_responses}")

    # Create batch tensor
    max_len = max(len(response) for response in encoded_responses)
    padded_responses = [
        response + [tokenizer.pad_token_id] * (max_len - len(response)) for response in encoded_responses
    ]
    model_responses_tensor = torch.tensor(padded_responses)

    # Calculate rewards for entire batch
    rewards = reward_calculator(model_responses_tensor, correct_answers)
    print(f"Calculated rewards: {rewards}")
