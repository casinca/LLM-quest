import os

import torch

import config
from llm_quest.alignment.rlhf_grpo.grpo_engine import (
    GRPOEvaluator,
    batched_responses_collator,
    grpo_loss,
    kl_div_per_token,
    log_probs_per_token,
    z_scores,
)
from llm_quest.gpt.generate import generate_batched_loop
from llm_quest.utils import CheckpointEvaluator, ResponseExtractor


class PrefixMatchingReward:
    """
    Reward calculator following the prefix matching strategy from the RPT paper:
    see 3.2 Pre-Training with Reinforcement Learning and Appendix A. Design Choices of Reward

    2 conditions must be met to get a positive reward:
        - the answer must be a prefix of the labels
        - the answer must be a valid boundary of the labels

        The valid boundary is the set of all possible byte lengths of the labels.

    args:
        tokenizer: the tokenizer to use to tokenize the labels
        good_answer_reward: the reward for a good answer
        wrong_answer_reward: the penalty for a wrong answer
        unfinished_answer_reward: the penalty for an unfinished answer

    """

    def __init__(
        self,
        tokenizer,
        good_answer_reward=1.0,
        wrong_answer_reward=0.0,
        unfinished_answer_reward=-10.0,
    ):
        assert wrong_answer_reward <= 0, "wrong_answer_reward should be ≤ 0"
        assert unfinished_answer_reward <= 0, "unfinished_answer_reward should be ≤ 0"

        self.tokenizer = tokenizer
        self.good_answer_reward = good_answer_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.unfinished_answer_reward = unfinished_answer_reward

    @staticmethod
    def _is_prefix(answer_bytes, label_bytes):
        """
        Args:
            answer_bytes (bytes): The answer in bytes.
            label_bytes (bytes): The label in bytes.
        """
        return label_bytes.startswith(answer_bytes)

    @staticmethod
    def _is_valid_boundary(answer_bytes, valid_boundary):
        """
        Args:
            answer_bytes (bytes): The answer in bytes.
            valid_boundary (set[int]): The valid boundary of the labels.
        """
        return len(answer_bytes) in valid_boundary

    def _get_valid_boundary(self, label):
        """
        calc & return the valid boundary of the current label string.

        Args:
            label (str): The ground truth label.

        Returns:
            set[int]: The valid boundary of the label.
        """
        valid_boundary = set()
        token_ids = self.tokenizer.encode(label)

        for i in range(1, len(token_ids) + 1):
            token_id = token_ids[:i]
            token_string = self.tokenizer.decode(token_id)
            token_bytes = token_string.encode("utf-8")
            valid_boundary.add(len(token_bytes))

        return valid_boundary

    def _calc_reward(self, model_responses, labels):
        """
        Calculate the rewards based on the model's answers, for a batch.

        Args:
            model_responses (list[str]): The decoded model's responses.
            labels (list[str]): The ground truth labels.

        Returns:
            list[float]: The rewards for a batch.

        """
        rewards_list = []

        for response_string, label in zip(model_responses, labels):
            model_answer = ResponseExtractor.get_answer(response_string)

            if model_answer is None:
                rewards_list.append(self.unfinished_answer_reward)
                continue

            valid_boundary = self._get_valid_boundary(label)
            # convert to bytes before checking both conditions
            answer_bytes = model_answer.encode("utf-8")
            label_bytes = label.encode("utf-8")

            # check both conditions
            if (
                PrefixMatchingReward._is_prefix(answer_bytes, label_bytes)
                and PrefixMatchingReward._is_valid_boundary(answer_bytes, valid_boundary)
            ):  # fmt: skip
                rewards_list.append(self.good_answer_reward)
            else:
                rewards_list.append(self.wrong_answer_reward)

        return rewards_list

    def __call__(self, model_responses, labels):
        """
        Main orchestrator for the rewards' calculation.

        Args:
            model_responses (torch.Tensor): The model's responses, shape (B, S)
            labels (list[str]): The ground truth labels.

        Returns:
            torch.Tensor: The total rewards for a batch of responses, shape (B,)
        """
        decoded_responses = self.tokenizer.batch_decode(model_responses, skip_special_tokens=True)
        rewards_list = self._calc_reward(decoded_responses, labels)

        return torch.tensor(rewards_list, dtype=torch.bfloat16, device=model_responses.device)


def rpt_grpo_training_loop(
    train_loader,
    val_loader,
    policy_model,
    reference_model,
    tokenizer,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    max_gen=70,
    clip_eps=0.2,
    beta=1.0,
    evaluation=True,
    eval_freq=None,
    eval_batches=None,
    eval_num_samples=1,
    kl_div_threshold=0.5,
):
    """
    Reinforcement Pretraining (RPT) training loop with GRPO, derived from rlvr_grpo_training_loop().

    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (π_θ, also used for π_θ_old).
        reference_model (nn.Module): A copy of the policy model (as π_ref) used to compute:
                                    - KL divergence (D_KL(π_ref || π_θ)).
        tokenizer (Tokenizer): The tokenizer to decode the responses (needs a batch_decode method).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model for each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model (used for context length).
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        max_gen (int): Maximum number of tokens to generate for each response.
        clip_eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.
        evaluation (bool, optional): Whether to perform evaluation. Defaults to True.
        eval_freq (int, optional): Frequency (in training steps) at which to perform evaluation. Defaults to None.
        eval_batches (int, optional): Number of batches to evaluate on. If None, evaluates on the whole val_loader.
        eval_num_samples (int, optional): Number of responses to generate per prompt for evaluation. Defaults to 1.
        kl_div_threshold (float, optional): max KL divergence allowed for checkpoint saving. Defaults to 0.5.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reference_model.eval()
    reward_calculator = PrefixMatchingReward(tokenizer=tokenizer)
    chkp_eval = CheckpointEvaluator(kl_div_threshold=kl_div_threshold, min_reward_threshold=0.35, beta=beta)

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
            correct_answers = [ans for ans in batch["answers"] for _ in range(num_samples)]

            responses = generate_batched_loop(
                input_tensor=dup_prompts,
                model=policy_model,
                attention_mask=dup_prompts_masks,
                max_gen=max_gen,
                context_length=policy_config["context_length"],
                top_k=20,
                temp=1,
                last_real=last_real_pos,
            )  # responses 2D shape: (batch_size * num_samples, max_prompt_len + max_gen), for simplicity: (B, S)

            collated_batch = batched_responses_collator(
                responses,
                len_prompt=batch["padded_prompts"].shape[-1],
                device=device,
            )

            # --- Retrieving logprobs & rewards ---
            with torch.inference_mode():
                # why intermediate masking with loss_mask for logprobs : TODO
                loss_mask = collated_batch["reward_masks"][:, 1:]

                old_logprobs = log_probs_per_token(  # shape: (B, S-1)
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=loss_mask,
                )
                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=loss_mask,
                )

                rewards = reward_calculator(  # shape: (B,)
                    model_responses=collated_batch["padded_responses"],
                    labels=correct_answers,
                )

            advantages = z_scores(rewards, num_samples)  # grouping and computing zscores (outside the inference scope)

            # --- Gradient updates loop ---
            policy_model.train()
            cum_grpo_loss = 0.0

            for grad_step in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=loss_mask,
                )

                policy_ratio_per_token = torch.exp(policy_logprobs - old_logprobs)
                kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)  # (will be masked in the loss calc)

                # loss, backprop, update
                grpo_loss_batch = grpo_loss(
                    policy_ratio=policy_ratio_per_token,
                    advantages=advantages,
                    loss_mask=loss_mask,
                    min_clip=clip_eps,
                    max_clip=clip_eps,
                    beta=beta,
                    kl_div=kl_div,
                    num_samples=num_samples,
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
                    evaluation_type="rlvr",
                    reward_calculator=reward_calculator,
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

                ## save new best checkpoint
                # if chkp_eval.is_rlvr_grpo_best(eval_metrics["val_kl_div"], eval_metrics["val_reward"]):
                #    save_path = os.path.join(
                #        config.rlvr_grpo_checkpoint_dir,
                #        f"best_checkpoint_{step}_score_{chkp_eval.max_score_grpo:.3f}.pt",
                #    )
                #    torch.save(policy_model.state_dict(), save_path)
