import torch

from llm_quest.alignment.grpo.grpo_engine import (
    GRPOEvaluator,
    batched_responses_collator,
    kl_div_per_token,
    log_probs_per_token,
    z_scores,
)
from llm_quest.gpt.generate import generate_batched_loop
from llm_quest.utils import CheckpointEvaluator, ResponseExtractor


class VerifiableRewardCalculator:
    """
    Reward calculator for a batch of responses, based on simple heuristics.

    Args:
        tokenizer (Tokenizer): The tokenizer to decode the responses (needs a batch_decode method).
        answer_reward_value (float): The reward value for a correct answer.
        answer_penalty (float): The penalty for an incorrect answer (should be >= 0).
        reasoning_weight (float): A coeff to weight the reasoning reward vs. the answer.
        pad_token_id (int): The token id to use for padding.

    """

    def __init__(
        self,
        tokenizer,
        answer_reward_value=10.0,
        answer_penalty=0.0,
        reasoning_weight=0.0,
        pad_token_id=50256,  # NOTE: placeholder for now, in case we need to penalize for unfinished responses
    ):
        assert answer_penalty >= 0, "answer_penalty should be >= 0"

        self.tokenizer = tokenizer
        self.answer_reward_value = answer_reward_value
        self.answer_penalty = answer_penalty
        self.reasoning_weight = reasoning_weight  # placeholder for now in case I want to do something fancy

    def _calc_answer(self, response_strings, correct_answers):
        """
        Calculate the rewards of the model's answers for a batch.

        Args:
            response_strings (list): The model's answers.
            correct_answers (list): The correct answers.

        Returns:
            list: The rewards of the model's answer for a batch.

        """
        rewards_list = []

        for response_string, correct_answer in zip(response_strings, correct_answers):
            model_answer = ResponseExtractor.get_answer(response_string)

            if model_answer is not None and float(model_answer) == float(correct_answer):
                rewards_list.append(self.answer_reward_value)
            else:
                rewards_list.append(-self.answer_penalty)

        return rewards_list

    def _calc_reasoning(self):
        pass
        # NOTE: For now, following DeepSeek and AI2 TULU's impl, returning the answer only

    def __call__(self, model_responses, correct_answers):
        """
        Main orchestrator for the rewards' calculation.

        Args:
            model_responses (torch.Tensor): The model's responses.
            correct_answers (list): The correct answers.

        Returns:
            torch.Tensor: The total rewards for a batch of responses, shape (batch_size,)
            (total_rewards = answer_rewards)

        """
        decoded_strings = self.tokenizer.batch_decode(model_responses, skip_special_tokens=False)

        answer_rewards = self._calc_answer(decoded_strings, correct_answers)

        return torch.tensor(
            answer_rewards, dtype=model_responses.dtype, device=model_responses.device
        )  # shape (batch_size,)


def rlvr_grpo_prompt_collator(batch, pad_token_id=50256, custom_max_length=None, device="cpu"):
    """
    Collate function to pad prompts of different lengths into a single tensor, preparing them for the policy model
    sample generations. It also passes through the answers kept as strings.
    This is a slight variation from the original rlhf_grpo_prompt_collator() to return the answers.

    Args:
        batch (List[Dict[str, any]]): A list of samples from ReasoningDataset, each a dict with "prompt" and "answer".
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        custom_max_length (int, optional): Maximum length of the padded sequences. If None, the maximum length
                is determined by the longest prompt in the batch.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cpu".

    Returns:
        Dict[str, torch.Tensor or list]: A dictionary containing:
            padded_prompts: Tensor of shape (batch_size, max_len) with padded prompt token IDs.
            prompt_masks: Boolean tensor of the same shape to keep track of padded tokens.
            last_real_pos: Tensor of shape (batch_size,) containing the position of the last real token in each prompt.
            answers: List of answer strings.
    """
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]

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
    tokenizer,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    max_gen=70,
    eps=0.2,
    beta=1.0,
    evaluation=True,
    eval_freq=None,
    eval_batches=None,
    eval_num_samples=1,
    kl_div_threshold=0.5,
):
    """
    GRPO training loop.

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
        eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
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

    reward_calculator = VerifiableRewardCalculator(tokenizer=tokenizer)
    reference_model.eval()
    chkp_eval = CheckpointEvaluator(kl_div_threshold, rlhf_min_score_threshold=0.0, rlhf_beta=beta)

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
                # Reward model - retrieving advantages -
                # full reward = mean pooling over the sequence length (I chose Outcome Supervision, ie not per token)
                rewards = reward_calculator(  # shape: (B,)
                    model_responses=collated_batch["padded_responses"],
                    correct_answers=correct_answers,
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

                # --- GRPO loss ---
                # (PyTorch will broadcast the advantages anyway, unsqueezing to emphasize advantages aren't per tokens)
                # ie, each trajectory gets a single advantage.
                surr_obj_per_token = policy_ratio_per_token * advantages.unsqueeze(-1)  # shapes (B,S-1) * (B,1)
                clipped_surr_obj_per_token = torch.clip(
                    policy_ratio_per_token, min=1 - eps, max=1 + eps
                ) * advantages.unsqueeze(-1)

                grpo_loss_per_token = -(torch.min(surr_obj_per_token, clipped_surr_obj_per_token) - beta * kl_div)
                grpo_loss_per_token *= loss_mask  # final masking: prompt + padding tokens
                grpo_loss = grpo_loss_per_token.sum() / loss_mask.sum()

                optimizer.zero_grad()
                grpo_loss.backward()
                optimizer.step()

                cum_grpo_loss += grpo_loss.item()

            avg_grpo_loss = cum_grpo_loss / num_grad_updates

            # --- Evaluation ---
            if evaluation and (step % eval_freq == 0):
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
                # if chkp_eval.is_rlhf_grpo_best(eval_metrics["val_kl_div"], eval_metrics["val_reward"]):
                #    save_path = os.path.join(
                #        config.rlhf_grpo_checkpoint_dir, f"best_checkpoint_{step}_score_{chkp_eval.max_score:.3f}.pt"
                #    )
                #    torch.save(policy_model.state_dict(), save_path)


# quick test
if __name__ == "__main__":
    import transformers

    string_response = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "<think>Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.</think>\n <answer>72 </answer><|endoftext|>",
    )
    correct_answer = "72.0"

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    reward_calculator = VerifiableRewardCalculator(tokenizer=tokenizer)

    encoded_response = tokenizer.encode(string_response[1])
    print(f"Encoded response: {encoded_response}")
    model_responses_tensor = torch.tensor([encoded_response])

    rewards = reward_calculator(model_responses_tensor, [correct_answer])
    print(f"Calculated rewards: {rewards}")
