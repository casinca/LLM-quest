import tiktoken
import torch
import torch.nn.functional as F

import config
from gpt_download import download_and_load_gpt2
from llm_quest.gpt.generate import generate_loop
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import ids_to_text, load_weights_into_gpt, text_to_ids


def bt_loss(chosen_logits, rejected_logits, beta=1.0):
    """
    Compute the Bradley-Terry loss between chosen and rejected logits.
    This is the equivalent of the BCE loss, see dpo README equations 8 and 9.

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


# TODO update docstring
def reward_model_training_eval_loop_simple(
    train_loader,
    val_loader,
    reward_model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_iter=None,
    device=None,  # not used here, as the collate func is moving to the device
    beta=0.1,
):
    """
    A simple training and evaluation loop for a model.

    Args:
        train_loader (DataLoader): DataLoader containing training data batches
        val_loader (DataLoader): DataLoader containing validation data batches
        reward_model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates
        num_epoch (int): Number of epochs to train for
        eval_freq (int): Number of steps between evaluations
        eval_iter (int): Number of batches to use during evaluation
        device (torch.device): Device to run training on (cuda/cpu)
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

            pref_mini_rewards = reward_model(batch["chosen"])
            rej_mini_rewards = reward_model(batch["rejected"])

            pref_mask = batch["chosen_mask"].unsqueeze(-1)  # shape (b, s) → (b, s, 1)
            rej_mask = batch["rejected_mask"].unsqueeze(-1)

            # masking the rewards to the valid tokens
            pref_mini_rewards *= pref_mask
            rej_mini_rewards *= rej_mask

            # --- mean pooling over the sequence length ---
            num_valid_pref_tokens = pref_mask.sum(dim=1)  # we want to divide by the number of valid tokens
            num_valid_rej_tokens = rej_mask.sum(dim=1)
            pref_rewards = pref_mini_rewards.sum(dim=1) / num_valid_pref_tokens  # shape (b,1)
            rej_rewards = rej_mini_rewards.sum(dim=1) / num_valid_rej_tokens

            # shape (b,1) → (b)
            pref_rewards, rej_rewards = pref_rewards.squeeze(-1), rej_rewards.squeeze(-1)

            loss = bt_loss(pref_rewards, rej_rewards)

            loss.backward()
            optimizer.step()

            interval_train_loss += loss.item()
            interval_correct_pred += (pref_rewards > rej_rewards).sum().item()
            interval_train_count += pref_rewards.shape[0]

            # eval
            if step % eval_freq == 0:
                # Calculate metrics for the interval
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

                interval_train_loss = 0.0
                interval_correct_pred = 0
                interval_train_count = 0

    return tracking


def evaluate_reward_model(val_loader, reward_model):
    """
    Evaluate the reward model on the validation set.
    """
    total_loss = 0.0
    correct = 0
    count = 0

    reward_model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            pref_mini_rewards = reward_model(batch["chosen"])
            rej_mini_rewards = reward_model(batch["rejected"])

            pref_mask = batch["chosen_mask"].unsqueeze(-1)
            rej_mask = batch["rejected_mask"].unsqueeze(-1)

            pref_reward = (pref_mini_rewards * pref_mask).sum(dim=1) / pref_mask.sum(dim=1)
            rej_reward = (rej_mini_rewards * rej_mask).sum(dim=1) / rej_mask.sum(dim=1)

            loss = bt_loss(pref_reward.squeeze(-1), rej_reward.squeeze(-1))
            total_loss += loss.item()

            # count the number of correct predictions
            correct += (pref_reward > rej_reward).sum().item()
            count += pref_reward.shape[0]

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / count

    reward_model.train()
    return avg_loss, avg_acc


def response_collator(responses, len_prompt, pad_token_id=50256, device="cuda"):
    """
    Collate sampled responses into a single tensor, preparing them for the reward model.

    Args:
        responses (List[List[int]]): list of responses, each response is the prompt + the policy's output
        len_prompt (int): Length of the prompt portion to distinguish between prompt and policy's output tokens.
        pad_token_id (int, optional): Token ID to use for padding sequences. Defaults to 50256.
        device (str, optional): Device where the resulting tensors will be placed. Defaults to "cuda".

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            padded_responses: Tensor of shape (batch_size, max_len) with padded token IDs.
            reward_masks: Boolean tensor of the same shape with masked: prompt + padding tokens.
            attn_masks: Boolean tensor of the same shape with masked: padding tokens.
    """
    max_len = max(len(response) for response in responses)
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


if __name__ == "__main__":
    settings, params = download_and_load_gpt2(model_size="124M", models_dir=config.openai_pretrained_w_gpt2)

    tokenizer = tiktoken.get_encoding("gpt2")
    model_settings = config.config_creator("gpt_s")
    torch.manual_seed(123)

    device = "cuda"
    model = GPTModel(model_settings)
    model.eval()

    load_weights_into_gpt(model, params)

    model.to(device)  # we move the model to GPU *after* loading weights

    num_samples = 5
    responses = []

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

    responses = [
        [20, 21, 22, 50, 51, 5250, 51, 52],
        [20, 21, 22, 60, 61, 62],
        [20, 21, 22, 70, 71, 7270, 71, 7270, 71, 72],
        [20, 21, 22, 80, 81, 82],
        [20, 21, 22, 90, 91, 922, 90],
    ]

    print(text_to_ids("This is where it", tokenizer=tokenizer))

    collated_batch = response_collator(
        responses,
        len_prompt=3,
        pad_token_id=50256,
        device="cuda",
    )
    print(collated_batch["padded_responses"])
    print(collated_batch["reward_masks"])
    print(collated_batch["attn_masks"])
