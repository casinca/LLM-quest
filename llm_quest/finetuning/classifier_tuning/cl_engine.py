import torch

from llm_quest.engine import evaluate, global_loss


def accuracy_loader(data_loader, model, device, attn_mask=None):
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

    for X, y, attn_mask in data_loader:
        X = X.to(device)
        y = y.to(device)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(X, last_token_only=True, attn_mask=attn_mask)  # (b, num_classes)

            # Get highest preds for entire batch of sequence
            batch_preds = torch.argmax(logits, dim=-1)  # shape (b,)
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
    lr_scheduler,
    eval_freq,
    eval_iter,
    device,
):
    """
    Same function as training_eval_loop() modified for classification:
        - global_loss(classification=True) ie calc loss only for the last token of each sequence
    """

    step = 0

    # keeping record of metrics for plotting
    train_losses, val_losses, train_accus, val_accus = [], [], [], []

    for epoch in range(1, num_epoch + 1):
        model.train()
        for input_batch, targets, attention_mask in train_loader:
            input_batch = input_batch.to(device)
            targets = targets.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(
                input_batch, last_token_only=True, attn_mask=attention_mask
            )  # only interested in the last tokens' logits for our classification (dirty - either pad or valid token)

            optimizer.zero_grad()
            loss = global_loss(logits, targets, model, classification=True)  # classi=True for correct loss req shape
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            lr_scheduler.step(step)
            optimizer.step()
            step += 1

            # eval
            if step == 1 or step % eval_freq == 0:
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device, classification=True)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                    f"lr: {lr_scheduler.current_lr:.1e}",
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
