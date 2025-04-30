import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from gpt_download import download_and_load_gpt2
from llm_quest.dataset import PreferenceDataset
from llm_quest.gpt.generate import generate_loop
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import ids_to_text, load_weights_into_gpt, text_to_ids


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


# TODO update docstring + change reshaping rewards instead of masks
def reward_model_training_eval_loop_simple(
    train_loader,
    val_loader,
    reward_model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_iter=None,
    device=None,
    beta=0.1,
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
        device (torch.device, optional): Device to run training on (e.g., 'cuda', 'cpu').
                Note: This parameter is currently not used internally as data loading handles device placement.
        beta (float, optional): Scaling factor for the Bradley-Terry loss. Defaults to 0.1.
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

    for epoch in range(1, num_epoch + 1):
        reward_model.train()

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            # shape (b, s, 1) → (b, s)
            pref_mini_rewards = reward_model(batch["chosen"]).squeeze(-1)
            rej_mini_rewards = reward_model(batch["rejected"]).squeeze(-1)

            pref_mask = batch["chosen_mask"]
            rej_mask = batch["rejected_mask"]

            # masking the rewards to the valid tokens
            pref_mini_rewards *= pref_mask
            rej_mini_rewards *= rej_mask

            # --- mean pooling over the sequence length ---
            num_valid_pref_tokens = pref_mask.sum(dim=1)  # we want to divide by the number of valid tokens
            num_valid_rej_tokens = rej_mask.sum(dim=1)
            pref_rewards = pref_mini_rewards.sum(dim=1) / num_valid_pref_tokens
            rej_rewards = rej_mini_rewards.sum(dim=1) / num_valid_rej_tokens

            loss = bt_loss(pref_rewards, rej_rewards)

            loss.backward()
            optimizer.step()

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
                val_loss, val_acc = evaluate_reward_model(val_loader, reward_model)
                tracking["val_losses"].append(val_loss)
                tracking["val_acc"].append(val_acc)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {avg_interval_train_loss:.5f}",
                    f"Train acc: {avg_interval_train_acc:.5f}",
                    f"Val loss: {val_loss:.5f}",
                    f"Val acc: {val_acc:.5f}",
                )

                # reset interval training metrics
                interval_train_loss = 0.0
                interval_correct_pred = 0
                interval_train_count = 0

    return tracking


# TODO might want to customize evaluation on a customizable number of batches
def evaluate_reward_model(val_loader, reward_model):
    """
    Evaluate the reward model on the full validation set.

    Args:
        val_loader (DataLoader): DataLoader providing batches of chosen and rejected responses.
        reward_model (nn.Module): The reward model being evaluated.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the validation set.
            - avg_acc (float): The average accuracy over the validation set
            (proportion of batches where chosen reward > rejected reward).
    """

    total_loss = 0.0
    correct = 0
    count = 0

    reward_model.eval()
    with torch.inference_mode():
        for batch in val_loader:

            pref_mini_rewards = reward_model(batch["chosen"]).squeeze(-1)
            rej_mini_rewards = reward_model(batch["rejected"]).squeeze(-1)

            pref_mask = batch["chosen_mask"]
            rej_mask = batch["rejected_mask"]

            pref_reward = (pref_mini_rewards * pref_mask).sum(dim=1) / pref_mask.sum(dim=1)
            rej_reward = (rej_mini_rewards * rej_mask).sum(dim=1) / rej_mask.sum(dim=1)

            loss = bt_loss(pref_reward, rej_reward)
            total_loss += loss.item()

            # count the number of correct predictions
            correct += (pref_reward > rej_reward).sum().item()
            count += pref_reward.shape[0]

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / count

    reward_model.train()
    return avg_loss, avg_acc


def grpo_prompt_collator(prompts, pad_token_id=50256, custom_max_length=None, device="cpu"):
    """
    Initial Collate function to pad prompts into a single tensor, preparing them for the policy model sample generations.

    Args:
        prompts (List[Dict[str, List[int]]]): A list of dictionaries, the dictionary must contain a key "prompt"
                whose value is a list of token IDs (int).
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        custom_max_length (int, optional): Maximum length of the padded sequences. If None, the maximum length
                is determined by the longest prompt in the batch.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cpu".

        Dict[str, torch.Tensor]: A dictionary containing:
            padded_prompts: Tensor of shape (batch_size, max_len) with padded prompt token IDs.
            prompt_masks: Boolean tensor of the same shape to keep track of padded tokens.
    """

    max_length = max(len(item["prompt"]) + 1 for item in prompts)
    if custom_max_length is not None:
        max_length = min(max_length, custom_max_length)

    padded_prompts = []
    prompt_masks = []

    for item in prompts:
        prompt_len = len(item["prompt"])
        padded_prompt = item["prompt"] + [pad_token_id] * (max_length - prompt_len)
        prompt_mask = [True] * prompt_len + [False] * (max_length - prompt_len)

        padded_prompts.append(padded_prompt)
        prompt_masks.append(prompt_mask)

    padded_prompts = torch.tensor(padded_prompts)
    prompt_masks = torch.tensor(prompt_masks, dtype=torch.bool)

    return {
        "padded_prompts": padded_prompts.to(device),
        "prompt_masks": prompt_masks.to(device),
    }


def response_collator(responses, len_prompt, pad_token_id=50256, device="cuda"):
    """
    Intended for use with grpo_training_loop_single_prompt().
    Collate sampled responses of different lengths into a single tensor, preparing them for the reward model.

    Args:
        responses (List[List[int]]): list of responses: each response is the prompt + the policy's output
        len_prompt (int): Length of the prompt portion to distinguish between prompt and policy's output tokens.
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cuda".

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            padded_responses: Tensor of shape (batch_size, max_len) with padded token IDs.
            reward_masks: Boolean tensor of the same shape with masked: prompt + padding tokens.
            attn_masks: Boolean tensor of the same shape with masked: padding tokens.
    """
    max_len = max(len(response) + 1 for response in responses)
    padded_responses = []
    reward_masks = []
    attn_masks = []

    for response in responses:
        len_response = len(response)  # len of a sampled response (ie prompt+policy's output)
        response += [pad_token_id] * (max_len - len_response)
        reward_mask = [False] * len_prompt + [True] * (len_response - len_prompt) + [False] * (max_len - len_response)
        attn_mask = [True] * (len_response) + [False] * (max_len - len_response)

        padded_responses.append(response)
        reward_masks.append(reward_mask)
        attn_masks.append(attn_mask)

    padded_responses = torch.tensor(padded_responses)
    reward_masks = torch.tensor(reward_masks, dtype=torch.bool)
    attn_masks = torch.tensor(attn_masks, dtype=torch.bool)

    return {
        "padded_responses": padded_responses.to(device),
        "reward_masks": reward_masks.to(device),
        "attn_masks": attn_masks.to(device),
    }


# TODO here responses are naturally padded with max_gen from generate_loop() but we can do a new generate function
# with variable response length and dynamic padding to go with this func.
def batched_response_collator(responses, len_prompt, pad_token_id=50256, device="cuda"):
    """
    Prepare batched sampled responses for the reward model.

    Since we are generating responses in parallel from generate_loop(), we don't need to pad anything, the generated
    tensor is already padded (with max_gen argument), we just need to prepare the masks.

    Args:
        responses (torch.Tensor): shape (batch_size, max_gen) responses = prompt + policy's output as padded token IDs.
        len_prompt (int): Length of the prompt portion to distinguish between prompt and policy's output tokens.
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cuda".

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            padded_responses: shape (batch_size, max_gen) responses = prompt + policy's output as padded token IDs.
            reward_masks: Boolean tensor of the same shape with masked: prompt + padding tokens.
            attn_masks: Boolean tensor of the same shape with masked: padding tokens.
    """
    b, max_len = responses.shape

    attn_masks = torch.ones(b, max_len, dtype=torch.bool)
    attn_masks = responses != pad_token_id

    reward_masks = torch.ones(b, max_len, dtype=torch.bool)
    reward_masks[:, :len_prompt] = False
    reward_masks[:, len_prompt:] = responses[:, len_prompt:] != pad_token_id

    return {
        "padded_responses": responses.to(device),
        "reward_masks": reward_masks.to(device),
        "attn_masks": attn_masks.to(device),
    }


def z_score(rewards):
    """
    Compute the z-score of the masked rewards.
    This is the way DeepSeek calculate the advantages.

    Args:
        rewards (torch.Tensor): Tensor of shape (batch_size,) containing the masked rewards.

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing the z-scores.
    """
    return (rewards - rewards.mean()) / rewards.std()


def log_probs_per_token(logits, inputs, attention_mask=None):
    """
    Compute and retrieve the log probabilities assigned to each label in a sequence.
    This is similar to the static method compute_logprobs() in the DPOLoss class.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, seq_len, vocab_size) containing the logits.
        inputs (torch.Tensor): Tensor of shape (batch_size, seq_len) containing the generated tokens.
        attention_mask (torch.Tensor, optional): Tensor of shape (batch_size, seq_len) containing the attention
                                                mask (ie, padded tokens are masked). Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_len-1) containing the log probabilities.
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

    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        label_log_probs *= shifted_mask

    return label_log_probs  # shape (b, s-1)


def kl_div_per_token(policy_logprobs, reference_logprobs):
    """
    Compute the KL divergence per token between the policy and reference log probabilities.
    Estimated with (Schulman, 2020) unbiased estimator:
    D_KL(π_θ || π_ref) =
    π_ref(y_i,t | x_i, y_i,<t) / π_θ(y_i,t | x_i, y_i,<t) -
    log(π_ref(y_i,t | x_i, y_i,<t) / π_θ(y_i,t | x_i, y_i,<t)) - 1

    Args:
        policy_logprobs (torch.Tensor): Tensor of shape (batch_size, seq_len) containing the policy log probabilities.
        reference_logprobs (torch.Tensor): Tensor of shape (batch_size, seq_len) containing the reference log
        probabilities.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, seq_len) containing the KL divergence per token.
    """
    ratio = torch.exp(reference_logprobs - policy_logprobs)
    log_ratio = reference_logprobs - policy_logprobs

    return ratio - log_ratio - 1


def grpo_training_loop_single_prompt(
    train_loader,
    policy_model,
    reference_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    eps=0.2,
    beta=1.0,
):
    """
    Earliest version for reference, not true to the GRPO paper.
    This GRPO training loop generates multiple samples for a single prompt at a time.
    Ie, it will only work for batch size = 1.


    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (policy π_θ).
        reference_model (nn.Module): A copy of the policy model (π_ref/π_θ_old) used to compute:
                                    - KL divergence (D_KL(π_ref || π_θ)).
                                    - Policy ratio (π_θ/π_ref).
                                    Its weights are periodically synchronized with the policy model.
        reward_model (nn.Module): A model (r_𝜑) pretrained to predict rewards for completions (frozen).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model from each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model.
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()

    for epoch in range(1, num_epoch + 1):

        for batch in train_loader:
            print("start")
            reference_model.load_state_dict(policy_model.state_dict())
            reference_model.eval()
            policy_model.eval()
            responses = []
            # note: generate_loop() comes with torch.inference_mode(), no need to reapply here

            # --- Sampling responses ---
            # simple sequential sampling for single prompt
            for i in range(num_samples):
                torch.manual_seed(123 + i)

                response = generate_loop(
                    input=batch["padded_prompts"],
                    model=policy_model,
                    max_gen=20,
                    context_length=policy_config["context_length"],
                    top_k=25,
                    temp=1.4,
                )
                responses.append(response.squeeze(0).tolist())

            collated_batch = response_collator(
                responses,
                len_prompt=batch["padded_prompts"].shape[-1],
                pad_token_id=50256,
                device=device,
            )

            with torch.inference_mode():
                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )
                # --- Reward model - retrieving advantages ---
                # full reward = mean pooling over the sequence length (I chose Outcome Supervision, ie not per token)
                mini_rewards = reward_model(
                    collated_batch["padded_responses"],
                    collated_batch["attn_masks"],
                ).squeeze(-1)
                mini_rewards *= collated_batch["reward_masks"]
                rewards = mini_rewards.sum(dim=1) / collated_batch["reward_masks"].sum(dim=1)

            advantages = z_score(rewards)  # computing outside the inference scope for grads

            # --- Gradient updates loop ---
            policy_model.train()
            for j in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )

                # πθ(y_i,t | x_i, y_i,<t) / π_ref(y_i,t | x_i, y_i,<t) =
                # exp(log(y_i,t | x_i, y_i,<t)) - log(π_ref(y_i,t | x_i, y_i,<t)))
                policy_ratio_per_token = torch.exp(policy_logprobs - reference_logprobs)

                # --- GRPO loss ---
                # KL divergence per token (could have also been reparametrized in terms of policy_ratio)
                kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)
                # (PyTorch will broadcast the advantages, unsqueeze to emphasize advantages aren't per tokens)
                surrogate_loss_per_token = policy_ratio_per_token * advantages.unsqueeze(-1)
                clipped_surrogate_loss_per_token = torch.clip(
                    policy_ratio_per_token, min=1 - eps, max=1 + eps
                ) * advantages.unsqueeze(-1)

                grpo_loss_per_token = (
                    torch.min(surrogate_loss_per_token, clipped_surrogate_loss_per_token) - beta * kl_div
                )
                # masking prompt tokens from the policy ratio+kl_div (advantages was done only on rewards/responses)
                loss_mask = collated_batch["reward_masks"][:, 1:]
                grpo_loss_per_token *= loss_mask
                grpo_loss = grpo_loss_per_token.sum() / loss_mask.sum()

                optimizer.zero_grad()
                grpo_loss.backward()
                optimizer.step()


def deprecated_grpo_training_loop_variant(
    train_loader,
    policy_model,
    reference_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    eps=0.2,
    beta=1.0,
):
    """
    This was actually my initial GRPO training loop attempt, and wasn't true to the GRPO paper, I kept it for
    reference as:
        This update the reference model per episode/batch instead of the classic "per outer loop iteration" from the
        original paper. The rationale was that we could use a single model for both π_ref and π_θ_old.

    The drawback is that its anchor role isn't as strong, since the π_ref is updated every batch.



    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (policy π_θ).
        reference_model (nn.Module): A copy of the policy model (π_ref/π_θ_old) used to compute:
                                    - KL divergence (D_KL(π_ref || π_θ)).
                                    - Policy ratio (π_θ/π_ref).
                                    Its weights are periodically synchronized with the policy model.
        reward_model (nn.Module): A model (r_𝜑) pretrained to predict rewards for completions (frozen).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model from each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model.
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()

    for epoch in range(1, num_epoch + 1):

        for batch in train_loader:
            print("start")
            reference_model.load_state_dict(policy_model.state_dict())
            reference_model.eval()
            policy_model.eval()
            # note: generate_loop() comes with torch.inference_mode(), no need to reapply here

            torch.manual_seed(123)
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            # --- Sampling responses ---
            # simple sequential sampling for single prompt

            responses = generate_loop(
                input=dup_prompts,
                model=policy_model,
                max_gen=20,
                context_length=policy_config["context_length"],
                top_k=25,
                temp=0,
            )

            collated_batch = batched_response_collator(
                responses,
                len_prompt=batch["padded_prompts"].shape[-1],
                pad_token_id=50256,
                device=device,
            )

            with torch.inference_mode():
                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )
                # --- Reward model - retrieving advantages ---
                # full reward = mean pooling over the sequence length (I chose Outcome Supervision, ie not per token)
                mini_rewards = reward_model(
                    collated_batch["padded_responses"],
                    collated_batch["attn_masks"],
                ).squeeze(-1)
                mini_rewards *= collated_batch["reward_masks"]
                rewards = mini_rewards.sum(dim=1) / collated_batch["reward_masks"].sum(dim=1)

            advantages = z_score(rewards)  # computing outside the inference scope

            # --- Gradient updates loop ---
            policy_model.train()
            for j in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )

                # πθ(y_i,t | x_i, y_i,<t) / π_ref(y_i,t | x_i, y_i,<t) =
                # exp(log(y_i,t | x_i, y_i,<t)) - log(π_ref(y_i,t | x_i, y_i,<t)))
                policy_ratio_per_token = torch.exp(policy_logprobs - reference_logprobs)

                # --- GRPO loss ---
                # KL divergence per token (could have also been reparametrized in terms of policy_ratio)
                kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)
                # (PyTorch will broadcast the advantages, unsqueeze to emphasize advantages aren't per tokens)
                surrogate_loss_per_token = policy_ratio_per_token * advantages.unsqueeze(-1)
                clipped_surrogate_loss_per_token = torch.clip(
                    policy_ratio_per_token, min=1 - eps, max=1 + eps
                ) * advantages.unsqueeze(-1)

                grpo_loss_per_token = (
                    torch.min(surrogate_loss_per_token, clipped_surrogate_loss_per_token) - beta * kl_div
                )
                # masking prompt tokens from the policy ratio+kl_div (advantages was done only on rewards/responses)
                loss_mask = collated_batch["reward_masks"][:, 1:]
                grpo_loss_per_token *= loss_mask
                grpo_loss = grpo_loss_per_token.sum() / loss_mask.sum()

                optimizer.zero_grad()
                grpo_loss.backward()
                optimizer.step()


def grpo_training_loop_variant_experimental(
    train_loader,
    policy_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    eps=0.2,
    beta=1.0,
):
    """
    GRPO training with just 2 models (1 policy + 1 reward model).
    This experimental version is based on the assumption that we can use a single policy for both π_θ and π_θ_old from
    the grpo_training_loop() func implementation.
    From that assumption and mixed with the deprecated_grpo_training_loop_variant() where we used a single model for
    both π_ref and π_θ_old:

    We are left with a single model for π_θ, π_θ_old and π_ref.
    The drawback is that π_ref anchor role isn't as strong, since it's updated every episode/batch with π_θ_old.

    Args:
        train_loader (DataLoader): DataLoader providing batches of prompts.
        policy_model (nn.Module): The language model being trained (policy π_θ) also used as π_θ_old and π_ref (KL div).
        reward_model (nn.Module): r_𝜑 pretrained to predict rewards for completions (frozen).
        optimizer (torch.optim.Optimizer): Optimizer for updating the policy model's parameters.
        num_epoch (int): The total number of training epochs.
        num_samples (int): The number of responses/samples to generate from the policy model for each prompt.
        num_grad_updates (int): The number of gradient update steps to perform per batch of sampled data.
        policy_config (dict): Configuration dictionary for the policy model.
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()

    for epoch in range(1, num_epoch + 1):

        for batch in train_loader:
            print("start")
            policy_model.eval()  # for every new episode, π_θ, π_θ_old and π_ref are the same
            # note: generate_loop() comes with torch.inference_mode(), no need to reapply here

            torch.manual_seed(123)
            # --- Sampling responses ---
            # interleaving the prompts to generate multiple samples in parallel
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            responses = generate_loop(
                input=dup_prompts,
                model=policy_model,
                max_gen=20,
                context_length=policy_config["context_length"],
                top_k=25,
                temp=0,
            )

            collated_batch = batched_response_collator(
                responses,
                len_prompt=batch["padded_prompts"].shape[-1],
                pad_token_id=50256,
                device=device,
            )

            with torch.inference_mode():
                # we now use π_θ in inference mode for both π_ref and π_θ_old
                old_and_ref_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )
                # Reward model - retrieving advantages -
                # full reward = mean pooling over the sequence length (I chose Outcome Supervision, ie not per token)
                mini_rewards = reward_model(
                    collated_batch["padded_responses"],
                    collated_batch["attn_masks"],
                ).squeeze(-1)
                mini_rewards *= collated_batch["reward_masks"]
                rewards = mini_rewards.sum(dim=1) / collated_batch["reward_masks"].sum(dim=1)

            advantages = z_score(rewards)  # computing outside the inference scope

            policy_model.train()
            # --- Gradient updates loop ---
            for j in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )

                # --- GRPO loss ---
                # π_ref = π_θ_old
                policy_ratio_per_token = torch.exp(policy_logprobs - old_and_ref_logprobs)
                # KL divergence per token (could have also been reparametrized in terms of policy_ratio)
                kl_div = kl_div_per_token(policy_logprobs, old_and_ref_logprobs)
                # (PyTorch will broadcast the advantages, unsqueeze to emphasize advantages aren't per tokens)
                surrogate_loss_per_token = policy_ratio_per_token * advantages.unsqueeze(-1)
                clipped_surrogate_loss_per_token = torch.clip(
                    policy_ratio_per_token, min=1 - eps, max=1 + eps
                ) * advantages.unsqueeze(-1)

                grpo_loss_per_token = (
                    torch.min(surrogate_loss_per_token, clipped_surrogate_loss_per_token) - beta * kl_div
                )
                # masking prompt + padding tokens from the policy ratio+kl_div
                # (advantages were calculated on rewards/response only, ie we didn't count the prompt + padded tokens)
                # (log probs were calculated on prompt + policy's output only, ie we didn't count padded tokens)
                # the mask will naturally cancel out the prompt part + padded tokens to focus on tokens of interest.
                loss_mask = collated_batch["reward_masks"][:, 1:]
                grpo_loss_per_token *= loss_mask
                grpo_loss = grpo_loss_per_token.sum() / loss_mask.sum()

                optimizer.zero_grad()
                grpo_loss.backward()
                optimizer.step()


def grpo_training_loop(
    train_loader,
    policy_model,
    reference_model,
    reward_model,
    optimizer,
    num_epoch,
    num_samples,
    num_grad_updates,
    policy_config,
    device,
    eps=0.2,
    beta=1.0,
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
        policy_config (dict): Configuration dictionary for the policy model.
        device (torch.device or str): The device (e.g., 'cuda', 'cpu') to perform computations on.
        eps (float): Clipping parameter ϵ for the policy ratio in the PPO-like clipped objective function.
        beta (float): Coefficient 𝛽 for the KL divergence penalty term in the loss. Controls the
                    trade-off between maximizing reward and staying close to the reference policy.

    Returns:
        None: The function modifies the `policy_model` in place.

    """
    reward_model.eval()
    reference_model.eval()

    for epoch in range(1, num_epoch + 1):
        reference_model.load_state_dict(policy_model.state_dict())

        for batch in train_loader:
            print("start")
            policy_model.eval()  # for every new episode, π_θ and π_θ_old are the same
            # note: generate_loop() comes with torch.inference_mode(), no need to reapply here

            torch.manual_seed(123)
            # --- Sampling responses ---
            # interleaving the prompts to generate multiple samples in parallel
            dup_prompts = batch["padded_prompts"].repeat_interleave(num_samples, dim=0)
            responses = generate_loop(
                input=dup_prompts,
                model=policy_model,
                max_gen=20,
                context_length=policy_config["context_length"],
                top_k=25,
                temp=0,
            )

            collated_batch = batched_response_collator(
                responses,
                len_prompt=batch["padded_prompts"].shape[-1],
                pad_token_id=50256,
                device=device,
            )

            with torch.inference_mode():
                old_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )
                reference_logprobs = log_probs_per_token(
                    logits=reference_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )
                # Reward model - retrieving advantages -
                # full reward = mean pooling over the sequence length (I chose Outcome Supervision, ie not per token)
                mini_rewards = reward_model(
                    collated_batch["padded_responses"],
                    collated_batch["attn_masks"],
                ).squeeze(-1)
                mini_rewards *= collated_batch["reward_masks"]
                rewards = mini_rewards.sum(dim=1) / collated_batch["reward_masks"].sum(dim=1)

            advantages = z_score(rewards)  # computing outside the inference scope

            policy_model.train()
            # --- Gradient updates loop ---
            for j in range(num_grad_updates):
                policy_logprobs = log_probs_per_token(
                    logits=policy_model(collated_batch["padded_responses"], collated_batch["attn_masks"]),
                    inputs=collated_batch["padded_responses"],
                    attention_mask=collated_batch["attn_masks"],
                )

                # --- GRPO loss ---
                policy_ratio_per_token = torch.exp(policy_logprobs - old_logprobs)
                kl_div = kl_div_per_token(policy_logprobs, reference_logprobs)
                # (PyTorch will broadcast the advantages, unsqueeze to emphasize advantages aren't per tokens)
                surrogate_loss_per_token = policy_ratio_per_token * advantages.unsqueeze(-1)
                clipped_surrogate_loss_per_token = torch.clip(
                    policy_ratio_per_token, min=1 - eps, max=1 + eps
                ) * advantages.unsqueeze(-1)

                grpo_loss_per_token = (
                    torch.min(surrogate_loss_per_token, clipped_surrogate_loss_per_token) - beta * kl_div
                )
                # masking prompt + padding tokens from the policy ratio+kl_div
                # (advantages were calculated on rewards/response only, ie we didn't count the prompt + padded tokens)
                # (log probs were calculated on prompt + policy's output only, ie we didn't count padded tokens)
                # the mask will naturally cancel out the prompt part + padded tokens to focus on tokens of interest.
                loss_mask = collated_batch["reward_masks"][:, 1:]
                grpo_loss_per_token *= loss_mask
                grpo_loss = grpo_loss_per_token.sum() / loss_mask.sum()

                optimizer.zero_grad()
                grpo_loss.backward()
                optimizer.step()


# some test
if __name__ == "__main__":
    #    settings, params = download_and_load_gpt2(model_size="124M", models_dir=config.openai_pretrained_w_gpt2)
    #
    #    tokenizer = tiktoken.get_encoding("gpt2")
    #    model_settings = config.config_creator("gpt_s")
    #    torch.manual_seed(123)
    #
    #    device = "cuda"
    #    model = GPTModel(model_settings)
    #    model.eval()
    #
    #    load_weights_into_gpt(model, params)
    #
    #    model.to(device)  # we move the model to GPU *after* loading weights
    #
    #    num_samples = 5
    #    responses = []

    #    for i in range(num_samples):
    #        torch.manual_seed(123 + i)
    #        response = generate_loop(
    #            input=text_to_ids("This is where it", tokenizer=tokenizer),
    #            model=model,
    #            max_gen=20,
    #            context_length=model_settings["context_length"],
    #            top_k=25,
    #            temp=1.4,
    #        )
    #        responses.append(response.squeeze(0).tolist())

    responses = torch.tensor(
        [
            [20, 21, 22, 50, 50256, 50256],
            [20, 21, 22, 34, 61, 62],
            [20, 21, 22, 50, 24, 62],
            [40, 41, 50256, 70, 71, 50256],
            [40, 41, 50256, 80, 81, 83],
            [40, 41, 50256, 90, 91, 92],
        ]
    )

    collated_batch = batched_response_collator(
        responses,
        len_prompt=3,
        pad_token_id=50256,
        device="cuda",
    )

    # collated_batch = response_collator(
    #    responses,
    #    len_prompt=3,
    #    pad_token_id=50256,
    #    device="cuda",
    # )

#    print(collated_batch["padded_responses"])
#    print(collated_batch["reward_masks"])
#    print(collated_batch["attn_masks"])
#
#    device = "cuda"
#    torch.manual_seed(123)
#    reward_model_cfg = config.GPT_SMALL_CONFIG
#
#    reward_model = GPTModel(reward_model_cfg)
#    # changing the head to a single output linear layer: we want a scalar reward
#    reward_model.out = nn.Linear(reward_model_cfg["emb_dim"], 1)
#
#    # freeze model - make all layers non-trainable
#    reward_model.eval()
#
#    reward_model.to(device)
#
#    pref_mini_rewards = reward_model(collated_batch["padded_responses"], collated_batch["attn_masks"]).squeeze(-1)
#    pref_mini_rewards *= collated_batch["reward_masks"]
#    pref_rewards = pref_mini_rewards.sum(dim=1) / collated_batch["reward_masks"].sum(dim=1)
#
#    print(pref_rewards)
#
#    print(z_score(pref_rewards))
