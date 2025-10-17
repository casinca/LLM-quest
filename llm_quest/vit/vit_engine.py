import math

import torch

# This is mostly copy pasta for now, adapting the training_eval_loop() and classifier_training_eval_loop() functions
# from `engine.py` for ViT architecture.
# TODO Need to see for refactoring. Either we regroup loss+class logic per architecture (current) or we change to per
# type of tasks (gen vs class (ViT+ causal class))


# TODO We might want to add some norm and dropout if this goes bonkers for the ffn type
class ViTAdapter(torch.nn.Module):
    """
    Adapter/connector for ViT to LLM:
    - If the ViT projected dimension is different from the LLM embedding dimension, we need a gateway, ie a linear layer
        that will reproject the ViT output to match the LLM embedding dimension.
    - If the ViT projected dimension is the same as the LLM embedding dimension, we can still use the adapter to help
        for domain adaption. Ie better alignment of the 2 multidimensional spaces (vision and text) rather than leaving
        the LLM solo learn from the ViT output directly.

    The weights of the adapter are trained during Multi-modal SFT/VQA.

    Args:
        vit_d_out (int): The output dimension of the ViT model.
        llm_d_in (int): The input/embedding dimension of the LLM model.
        adapter_type (str): The type of adapter to use.
            - "simple": A simple linear layer.
            - "ffn": 1 hidden layer feed-forward neural network.
        hidden_size_factor (int): The expansion factor for the hidden layer of the FFN.
    """

    def __init__(self, vit_d_out, llm_d_in, adapter_type="simple", hidden_size_factor=4):
        super().__init__()

        if adapter_type == "simple":
            self.adapter = torch.nn.Linear(vit_d_out, llm_d_in)

        elif adapter_type == "ffn":
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(vit_d_out, vit_d_out * hidden_size_factor),
                torch.nn.GELU(),
                torch.nn.Linear(vit_d_out * hidden_size_factor, llm_d_in),
            )

        else:
            raise ValueError(f"Invalid adapter type: {adapter_type}")

    def forward(self, x):
        return self.adapter(x)


def vit_training_eval_loop(
    train_loader,
    val_loader,
    model,
    optimizer,
    num_epoch,
    warmup_percent,
    init_lr,
    peak_lr,
    min_lr,
    eval_freq,
    eval_iter,
    device,
    use_amp=True,
):
    """
    A training and evaluation loop for ViT with learning rate scheduler,
    gradient clipping, mixed precision training (AMP), and accuracy tracking.
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        model (torch.nn.Module): ViT model to train
        optimizer (torch.optim.Optimizer): Optimizer to use for training
        num_epoch (int): Number of epochs to train for
        warmup_percent (float): Percentage of total steps to use for learning rate warmup,
                                if set to 0.0, will disable warmup
        init_lr (float): Initial learning rate for warmup (value doesn't matter if warmup disabled)
        peak_lr (float): Peak learning rate after warmup
        min_lr (float): Minimum learning rate for cosine decay (if same as peak_lr, will disable decay)
        eval_freq (int): Number of steps between evaluations
        eval_iter (int): Number of batches to use during evaluation
        device (torch.device): Device to run training on (cuda/cpu)
        use_amp (bool): Whether to use Automatic Mixed Precision training
    Returns:
        tuple: A tuple containing:
            - train_losses (list): Training losses
            - val_losses (list): Validation losses
            - train_accus (list): Training accuracies
            - val_accus (list): Validation accuracies
    """
    step = -1
    total_steps = len(train_loader) * num_epoch
    warmup_steps = int(warmup_percent * total_steps)
    if warmup_percent and warmup_steps:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # Keep a record of metrics for plotting
    train_losses, val_losses, train_accus, val_accus = [], [], [], []

    for epoch in range(1, num_epoch + 1):

        model.train()

        for input_batch, targets in train_loader:
            step += 1

            # Learning rate scheduler logic:
            # lr update with warmup and cosine decay = 0.5 * (1 + cos(π * curr_step / total_step))
            # curr_step and total_step are steps after the warmup, thus needs to be adjusted for the warmup difference
            if step < warmup_steps:
                lr = init_lr + step * lr_increment
            else:
                decay_steps = total_steps - warmup_steps  # total step adjusted for warmup
                curr_step = step - warmup_steps  # curr decay step adjusted for warmup
                cosine_decay = 0.5 * (1 + math.cos(math.pi * curr_step / decay_steps))
                lr = min_lr + (peak_lr - min_lr) * cosine_decay

            # update lr
            for param_group in optimizer.param_groups:
                if not param_group.get("custom_lr", False):  # only adjust lr for non-custom groups
                    param_group["lr"] = lr

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            # Autocast enable/disable for mixed precision training
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_batch)
                loss = torch.nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad()

            # Backward pass and optimizer step (simplified - no scaler needed for bfloat16)
            loss.backward()
            # gradient clipping at a max norm of 1 (after warmup)
            if step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # eval (AMP disabled for evaluation with torch no_grad in evaluate())
            if step % eval_freq == 0:
                train_loss, val_loss = ViT.evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )

    # accuracy calc per epoch
    train_accu_epoch = ViT.accuracy_loader(train_loader, model, device)
    val_accu_epoch = ViT.accuracy_loader(val_loader, model, device)
    train_accus.append(train_accu_epoch)
    val_accus.append(val_accu_epoch)

    print(
        f"training accu epoch {epoch}: {train_accu_epoch*100:.4f}%",
        f"validation accu epoch {epoch}: {val_accu_epoch*100:.4f}%",
    )

    return train_losses, val_losses, train_accus, val_accus


class ViT:
    """
    Wrapper for evaluation functions adapted for Vision Transformer (ViT) model architecture.
    This class handles eval + accu calculations.
    """

    @staticmethod
    def accuracy_loader(data_loader, model, device):
        """Calculate accuracy of the ViT model's predictions on a data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader containing batches of images and labels
            model (torch.nn.Module): The ViT model to evaluate
            device (torch.device): Device to run evaluation on (cuda/cpu)

        Returns:
            float: Accuracy score between 0 and 1, representing fraction of correct predictions
        """
        model.eval()
        total_correct_preds, total_num_examples = 0, 0

        with torch.no_grad():
            for X, y in data_loader:

                X = X.to(device)
                y = y.to(device)

                logits = model(X)  # (b, num_classes)

                # get the highest predictions for the entire batch
                batch_preds = torch.argmax(logits, dim=-1)  # shape (b,)
                # compare with targets and sum the correct predictions
                total_correct_preds += (batch_preds == y).sum().item()
                total_num_examples += len(batch_preds)

        return total_correct_preds / total_num_examples

    @staticmethod
    def evaluate(train_loader, val_loader, model, eval_iter, device):
        """
        Evaluates the ViT model's performance on training and validation datasets.

        Args:
            train_loader (DataLoader): DataLoader containing the training data batches.
            val_loader (DataLoader): DataLoader containing the validation data batches.
            model (nn.Module): The ViT model to evaluate.
            eval_iter (int): Number of batches to use/iterate for evaluation.
            device (torch.device): The device to run evaluation on.

        Returns:
            tuple: A tuple containing:
                - train_loss (float): The average loss on the training dataset
                - val_loss (float): The average loss on the validation dataset
        """
        model.eval()
        with torch.no_grad():
            train_loss = ViT.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = ViT.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

        return train_loss, val_loss

    @staticmethod
    def calc_loss_loader(dataloader, model, device, num_batches=None):
        """
        Calculates the average loss across a customized number of batches from a dataloader
        for ViT classification tasks.

        Args:
            dataloader (DataLoader): The DataLoader containing the batches of data.
            model (nn.Module): The ViT model used to make predictions.
            device (torch.device): The device on which the model and tensors are located.
            num_batches (int, optional): The number of batches to evaluate. If None, evaluates all batches.

        Returns:
            float: The average loss across the specified number of batches. Returns NaN if dataloader is empty.
        """
        total_loss = 0

        # Check for smaller num of batches in order to speed up evaluation
        if len(dataloader) == 0:
            return float("NaN")
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        for i, (X, y) in enumerate(dataloader):
            if i < num_batches:
                loss = ViT._calc_loss_batch(X, y, model, device)
                total_loss += loss.item()

        # Return mean loss
        return total_loss / num_batches

    @staticmethod
    def _calc_loss_batch(X, y, model, device):
        """
        Calculates the CE loss for a batch of data for ViT classification.

        Args:
            X (torch.Tensor): The input image tensor for the batch (B, C, H, W).
            y (torch.Tensor): The target tensor for the batch (B,) with class indices.
            model (nn.Module): The ViT model used to make predictions.
            device (torch.device): The device on which the model and tensors are located.

        Returns:
            torch.Tensor: The loss for the batch.
        """
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)

        return loss
