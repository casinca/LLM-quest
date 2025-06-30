import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """
    This class implements Direct Preference Optimization (DPO) loss described in the paper and @rasbt's impl converted
    into a class. I also added back label smoothing for noisy labels.

    The loss is computed based on the log probabilities from a policy model and a reference
    model for both chosen and rejected responses.

    Args:
        beta (float): Temperature parameter controlling the sensitivity of the loss to differences in log
        probabilities.
        label_smoothing (float): Label smoothing parameter based on Eq. 3 https://ericmitchell.ai/cdpo.pdf \n
        In the case of noisy labels, we can simulate that imprecision by adding a small parameter ϵ to soften
        the proba distrib of the targets. Eg, you believe your dataset labels might be wrong ϵ proportion of the time.
        ie, P(true labels) = 1-ε instead of 1, with ϵ = (0, ..., 0.5)
        Disabled by default for original DPO behavior.
    """

    def __init__(self, beta=0.1, label_smoothing=0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def compute_logprobs(self, logits, inputs, attention_mask=None):
        """
        Computes the log probabilities for each token in a sequence based on the model's logits,
        optionally applying an attention mask (to avoid padding tokens getting calc'd)

        Args:
            logits (torch.Tensor): The output logits from the model, shape (b, s, v),
                                    where b is the batch size, s is the sequence length, and v is the vocabulary size.
            inputs (torch.Tensor): The input token IDs, shape (b, s).
            attention_mask (torch.Tensor, optional): A mask indicating which tokens should be considered.
                                                    Shape (b, s). Defaults to None.

        Returns:
            torch.Tensor: The average log probabilities of each sequence in the batch, shape (b).
        """
        # shifting (next-token prediction)
        logits = logits[:, :-1, :]  # adjusting logits, removing last token, shape (b, s-1, v)
        labels = inputs[:, 1:]  # labels are inputs shifted by 1, removing first token, shape (b, s-1)

        log_probs = F.log_softmax(logits, dim=-1)  # shape (b, s-1, v)

        # retrieving the log probs assigned to each label
        label_log_probs = torch.gather(
            input=log_probs,  # shape (b, s-1, v)
            dim=-1,
            # label shape (b, s-1) increasing dim as input & index must have the same dim (3D) → (b, s-1, 1)
            index=labels.unsqueeze(-1),
        )
        label_log_probs.squeeze_(-1)  # removing added dim, back to 2D shape (b, s-1)

        if attention_mask is not None:
            shifted_mask = attention_mask[:, 1:]  # shifting mask to match labels
            label_log_probs = label_log_probs * shifted_mask  # element wise multiplication to apply mask (prob*0=0)
            # returning mean with excluding masked/padded tokens
            avg_label_log_probs = label_log_probs.sum(-1) / attention_mask.sum(-1)

            return avg_label_log_probs  # shape (b,)

        else:
            return label_log_probs.mean(-1)

    def compute_loss(self, pol_chosen_logprob, pol_rejected_logprob, ref_chosen_logprob, ref_rejected_logprob):
        """
        Calculates the dpo loss for a batch. by comparing the policy model's output probabilities for chosen and
        rejected actions against those of the reference policy.\n
        Thus logprob tensors are of shape batch size (b), calculated from compute_logprobs()

        Args:
            pol_chosen_logprob (torch.Tensor): Log probabilities of the chosen actions from the policy model.
            pol_rejected_logprob (torch.Tensor): Log probabilities of the rejected actions from the policy model.
            ref_chosen_logprob (torch.Tensor): Log probabilities of the chosen actions from the reference policy.
            ref_rejected_logprob (torch.Tensor): Log probabilities of the rejected actions from the reference policy.

        Returns:
            tuple: (loss, chosen_rewards, rejected_rewards)
                - loss: The mean dpo loss for the batch
                - chosen_rewards: Mean reward for chosen responses
                - rejected_rewards: Mean reward for rejected responses
        """
        # log(πθ(y|x) / π_ref(y|x)) = log(πθ(y|x)) - log(π_ref(y|x))
        pref_logratio = pol_chosen_logprob - ref_chosen_logprob
        rejec_logratio = pol_rejected_logprob - ref_rejected_logprob
        # For logging: detach the ratios to prevent gradient computation
        chosen_rewards = pref_logratio.detach()
        rejected_rewards = rejec_logratio.detach()

        # logits in the sense raw scores that are passed to a sigmoid for probas
        logits = pref_logratio - rejec_logratio

        # plural because it's a vector of losses (per example/seq in the batch) shape (b)
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
        # Return averaged losses over a single batch and the detached rewards for logging
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def forward(self, batch, policy_model, reference_model):
        """
        returning dpo loss calc for a batch by combining compute_logprobs() and compute_loss()
        """

        pol_chosen_logprob = self.compute_logprobs(
            logits=policy_model(batch["chosen"]),
            inputs=batch["chosen"],
            attention_mask=batch["chosen_mask"],
        )
        pol_rejected_logprob = self.compute_logprobs(
            logits=policy_model(batch["rejected"]),
            inputs=batch["rejected"],
            attention_mask=batch["rejected_mask"],
        )

        with torch.no_grad():
            ref_chosen_logprob = self.compute_logprobs(
                logits=reference_model(batch["chosen"]),
                inputs=batch["chosen"],
                attention_mask=batch["chosen_mask"],
            )
            ref_rejected_logprob = self.compute_logprobs(
                logits=reference_model(batch["rejected"]),
                inputs=batch["rejected"],
                attention_mask=batch["rejected_mask"],
            )

        return self.compute_loss(
            pol_chosen_logprob,
            pol_rejected_logprob,
            ref_chosen_logprob,
            ref_rejected_logprob,
        )


class DPOEvaluator:
    def __init__(self, beta=0.1):
        self.beta = beta

    def _compute_dpo_loss_loader(self, dataloader, policy_model, reference_model, num_batches=None):
        """
        Computes the avg dpo loss for a given dataloader, over a specified number of batches.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
            policy_model (torch.nn.Module): The policy model being trained.
            reference_model (torch.nn.Module): The reference model for comparison.
            num_batches (int, optional): Number of batches to compute the loss for. Defaults to None

        Returns:
            tuple (float, float, float): (avg_total_loss, avg_total_chosen_rewards, avg_total_rejected_rewards)
        """
        if len(dataloader) == 0:
            return float("NaN")

        dpo_loss = DPOLoss(beta=self.beta)

        total_loss = total_chosen_rewards = total_rejected_rewards = 0.0
        # if num_batches is specified, only computing dpo loss over this num of batches
        num_batches = min(num_batches, len(dataloader)) if num_batches else len(dataloader)

        for i, batch in enumerate(dataloader):
            if i < num_batches:
                loss, chosen_rewards, rejected_rewards = dpo_loss.forward(batch, policy_model, reference_model)
                total_loss += loss.item()
                total_chosen_rewards += chosen_rewards.item()
                total_rejected_rewards += rejected_rewards.item()
            else:
                break

        avg_total_loss = total_loss / num_batches
        avg_total_chosen_rewards = total_chosen_rewards / num_batches
        avg_total_rejected_rewards = total_rejected_rewards / num_batches

        return avg_total_loss, avg_total_chosen_rewards, avg_total_rejected_rewards

    def evaluate(self, train_loader, val_loader, policy_model, reference_model, eval_iter=None):
        """
        Evaluate the policy model against the reference model on both training and validation sets.

        Args:
            policy_model (nn.Module): The policy model to evaluate
            reference_model (nn.Module): The reference model for comparison
            train_loader (DataLoader): DataLoader for the training dataset
            val_loader (DataLoader): DataLoader for the validation dataset
            eval_iter (int, optional): Number of batches to evaluate (optional)

        Returns:
            dict: Dictionary containing evaluation metrics for both training and validation sets
        """
        policy_model.eval()
        with torch.inference_mode():

            train_loss, train_chosen_rewards, train_rejected_rewards = self._compute_dpo_loss_loader(
                train_loader,
                policy_model,
                reference_model,
                num_batches=eval_iter,
            )

            val_loss, val_chosen_rewards, val_rejected_rewards = self._compute_dpo_loss_loader(
                val_loader,
                policy_model,
                reference_model,
                num_batches=eval_iter,
            )

            res = {
                "train_loss": train_loss,
                "train_chosen_reward": train_chosen_rewards,
                "train_rejected_reward": train_rejected_rewards,
                "val_loss": val_loss,
                "val_chosen_reward": val_chosen_rewards,
                "val_rejected_reward": val_rejected_rewards,
            }

        policy_model.train()

        return res


def dpo_training_eval_loop_simple(
    train_loader,
    val_loader,
    policy_model,
    reference_model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_iter,
    device,  # not used here, as the collate func is moving to the device
    beta=0.1,
):
    """
    A simple training and evaluation loop for a model.

    Args:
        train_loader (DataLoader): DataLoader containing training data batches
        val_loader (DataLoader): DataLoader containing validation data batches
        policy_model (nn.Module): The model to train
        reference_model (nn.Module): The reference model for comparison/baseline
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates
        num_epoch (int): Number of epochs to train for
        eval_freq (int): Number of steps between evaluations
        eval_iter (int): Number of batches to use during evaluation
        device (torch.device): Device to run training on (cuda/cpu)
    """
    step = 0
    # keeping track of metrics for plotting
    tracking = {
        "train_losses": [],
        "val_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
    }

    dpo_loss = DPOLoss(beta=beta)
    dpo_evaluator = DPOEvaluator(beta=beta)

    for epoch in range(1, num_epoch + 1):
        policy_model.train()

        for batch in train_loader:
            step += 1
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = dpo_loss.forward(batch, policy_model, reference_model)
            loss.backward()
            optimizer.step()

            # eval
            if step % eval_freq == 0:

                res = dpo_evaluator.evaluate(
                    train_loader,
                    val_loader,
                    policy_model,
                    reference_model,
                    eval_iter,
                )

                tracking["train_losses"].append(res["train_loss"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {res['train_loss']:.5f}, Val loss: {res['val_loss']:.5f}",
                    f"Train chosen reward: {res['train_chosen_reward']:.5f}",
                    f"Val chosen reward: {res['val_chosen_reward']:.5f}",
                )

    return tracking
