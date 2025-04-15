import math

import torch

from llm_quest.engine import evaluate, global_loss


def accuracy_loader(data_loader, model, device):
    """Calculate accuracy of the model's predictions on a data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader containing batches of input sequences and labels
        model (torch.nn.Module): The model to evaluate
        device (torch.device): Device to run evaluation on (cuda/cpu)

    Returns:
        float: Accuracy score between 0 and 1, representing fraction of correct predictions
    """

    model.eval()
    total_correct_preds, total_num_examples = 0, 0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            logits = model(X, only_last_token=True)  # (b, num_classes)

            # Get highest preds for entire batch of sequence
            batch_preds = torch.argmax(logits, dim=-1)  # shape (b,)
            print(batch_preds.shape)
            # compare with targets as boolean:
            # summing up the 1s(True) and then converting tensor to a scalar for calculation
            total_correct_preds += (batch_preds == y).sum().item()
            total_num_examples += len(batch_preds)

    return total_correct_preds / total_num_examples


def classifier_training_eval_loop(
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
):
    """
    Same function as training_eval_loop() modified for classification:
        - global_loss(classification=True) ie calc loss only for the last token of each sequence
        - slight edit to prevent error for no warmup and lr decay
    """

    step = -1
    total_steps = len(train_loader) * num_epoch

    warmup_steps = int(warmup_percent * total_steps)
    if warmup_percent > 0.0:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # keeping record of metrics for plotting
    train_losses, val_losses, train_accus, val_accus = [], [], [], []

    for epoch in range(1, num_epoch + 1):
        model.train()
        for input_batch, targets in train_loader:
            step += 1

            # lr update with warmup and cosine decay= 0.5 * (1 + cos(π * curr_step / total_step))
            # curr_step and total_step are steps after the warmup, thus needs to be adjusted for the warmup difference
            if step < warmup_steps:
                lr = init_lr + step * lr_increment
            else:
                decay_steps = total_steps - warmup_steps  # total step adjusted for warmup
                curr_step = step - warmup_steps  # curr decay step adjusted for warmup
                cosine_decay = 0.5 * (1 + math.cos(math.pi * curr_step / decay_steps))
                lr = min_lr + (peak_lr - min_lr) * cosine_decay

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            logits = model(input_batch, only_last_token=True)  # only interested in the last token/score for our classification

            optimizer.zero_grad()
            loss = global_loss(logits, targets, classification=True)  # classi=True for correct loss req shape
            loss.backward()

            # gradient clipping at a max norm of 1 (after warmup)
            if step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            # eval
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device, classification=True)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )

        # accuracy calc per epoch
        train_accu_epoch = accuracy_loader(train_loader, model, device)
        val_accu_epoch = accuracy_loader(val_loader, model, device)
        train_accus.append(train_accu_epoch)
        val_accus.append(val_accu_epoch)

        print(
            f"training accu epoch {epoch}: {train_accu_epoch*100:.4f}",
            f"validation accu epoch {epoch}: {val_accu_epoch*100:.4f}",
        )
