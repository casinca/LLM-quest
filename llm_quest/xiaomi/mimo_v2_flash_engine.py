# adapted copy pasta from deepseek_engine.py
# the training loop is simplified and more similar to the original training loop than training_eval_loop_mtp()
# from Deepseek. Because inputs for MTP modules are prepared/sliced on the fly inside the MiMoModel forward pass, this
# also means that we can reuse a simple collate function to prepare the data for the training loop.

import torch


def training_eval_loop_mimo(
    train_loader,
    val_loader,
    model,
    optimizer,
    num_epoch,
    lr_scheduler,
    eval_freq,
    eval_iter,
    device,
    use_amp=False,
):
    """
    Simple Training and evaluation loop for MiMo-V2-Flash model.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        model (torch.nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer to use for training
        num_epoch (int): Number of epochs to train for
        lr_scheduler (LearningRateScheduler): Learning rate scheduler object
        eval_freq (int): Number of steps between evaluations
        eval_iter (int): Number of batches to use during evaluation
        device (torch.device): Device to run training on (cuda/cpu)
        use_amp (bool): Whether to use Automatic Mixed Precision training
    """
    step = 0

    train_losses, val_losses = [], []

    for epoch in range(1, num_epoch + 1):
        model.train()

        # pretraining uses right padding, therefore attn masks are redundant with pytorch pad/no loss token id (-100)
        for input_batch, targets, _ in train_loader:
            input_batch = input_batch.to(device)
            targets = targets.to(device)

            # Autocast enable/disable for mixed precision training
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(input_batch, targets)

            optimizer.zero_grad()

            # Backward pass and optimizer step (simplified - no scaler needed for bfloat16)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            lr_scheduler.step(step)
            optimizer.step()
            step += 1

            # eval (AMP disabled for evaluation with torch no grad in evaluate())
            if step == 1 or step % eval_freq == 0:
                train_loss, val_loss = MiMoEvaluator.evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                    f"lr: {lr_scheduler.current_lr:.1e}",
                )

    return train_losses, val_losses


class MiMoEvaluator:
    """
    Wrapper of the evaluate() function and DSEvaluator, adapted for MiMo-V2-Flash model architecture.
    """

    @staticmethod
    def evaluate(train_loader, val_loader, model, eval_iter, device):

        model.eval()
        with torch.no_grad():
            train_loss = MiMoEvaluator.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = MiMoEvaluator.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

        return train_loss, val_loss

    @staticmethod
    def calc_loss_loader(dataloader, model, device, num_batches=None):

        total_loss = 0.0
        # checks for smaller num of batches in order to speed up evaluation
        if len(dataloader) == 0:
            return float("NaN")
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        # pretraining uses right padding, therefore attn masks are redundant with pytorch pad/no loss token id (-100)
        for i, (X, y, _) in enumerate(dataloader):
            if i < num_batches:
                X = X.to(device)
                y = y.to(device)
                # model returns loss when targets are provided (even in eval mode)
                loss = model(X, y)
                total_loss += loss.item()
        # returning mean
        return total_loss / num_batches
