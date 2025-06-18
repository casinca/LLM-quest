import math
import time

import torch


def training_eval_loop_mtp(
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
    Same as training_eval_loop() but adapted to DeepSeekV3 architecture and multi token prediction.
    A more robust training+eval loop with learning rate scheduler: warmup, decay (cosine), gradient clipping and
    mixed precision training (AMP).

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        model (torch.nn.Module): Model to train
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
    """
    step = -1
    total_steps = len(train_loader) * num_epoch
    warmup_steps = int(warmup_percent * total_steps)
    if warmup_percent:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # Keep a record of metrics for plotting.
    train_losses, val_losses = [], []

    for epoch in range(1, num_epoch + 1):
        model.train()

        for input_batch, targets, mtp_inputs, mtp_targets in train_loader:
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

            # Update lr
            for param_group in optimizer.param_groups:
                if not param_group.get("custom_lr", False):  # only adjust lr for non-custom groups
                    param_group["lr"] = lr

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            # Autocast enable/disable for mixed precision training
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(input_batch, targets, mtp_inputs, mtp_targets)

            optimizer.zero_grad()

            # Backward pass and optimizer step (simplified - no scaler needed for bfloat16)
            loss.backward()
            # gradient clipping at a max norm of 1 (after warmup)
            if step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # eval (AMP disabled for evaluation with torch no grad in evaluate())
            if step % eval_freq == 0:
                train_loss, val_loss = DS.evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )

    return train_losses, val_losses


def training_eval_loop_simple_timing(
    train_loader,
    val_loader,
    model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_iter,
    device,
):
    """
    A simple training and evaluation loop for a model tracking tok/s and memory alloc/res for local benchmarking.

    Args:
        train_loader (DataLoader): DataLoader containing training data batches
        val_loader (DataLoader): DataLoader containing validation data batches
        model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates
        num_epoch (int): Number of epochs to train for
        eval_freq (int): Number of steps between evaluations
        eval_iter (int): Number of batches to use during evaluation
        device (torch.device): Device to run training on (cuda/cpu)
    """
    step = 0
    last_tokens, total_tokens = 0, 0
    # keeping track of metrics for plotting
    train_losses, val_losses, track_tokens = [], [], []

    # Variables for cumulative average tokens/sec
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
        t_start.record()  # Start the timer for the first interval
    else:
        t0 = time.time()  # Start the timer for the first interval

    for epoch in range(1, num_epoch + 1):
        model.train()
        for input_batch, targets, mtp_inputs, mtp_targets in train_loader:
            step += 1

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = model(input_batch, targets, mtp_inputs, mtp_targets)
            loss.backward()
            optimizer.step()

            total_tokens += input_batch.numel()

            if step % eval_freq == 0:

                # End timing for the current interval
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Reset timer for the next interval

                # Calculate tokens processed in this interval
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                # Update cumulative counters (skip the first evaluation interval)
                if step > 1:  # This is False only when global_step == 0 (first evaluation)
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed

                # Compute cumulative average tokens/sec (excluding the first interval)
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # eval
                train_loss, val_loss = DS.evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                    f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}",
                )

        # Memory stats
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

            print(f"\nAllocated memory: {allocated:.4f} GB")
            print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


class DS:
    """
    Wrapper of the evaluate() function, in order to adapt for DeepSeek V3 model architecture and multi token prediction.
    """

    @staticmethod
    def evaluate(train_loader, val_loader, model, eval_iter, device):

        model.eval()
        with torch.no_grad():
            train_loss = DS.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = DS.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

        return train_loss, val_loss

    @staticmethod
    def calc_loss_loader(dataloader, model, device, num_batches=None):

        total_loss = 0
        # checks for smaller num of batches in order to speed up evaluation
        if len(dataloader) == 0:
            return float("NaN")
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(num_batches, len(dataloader))

        for i, (X, y, *_) in enumerate(dataloader):
            if i < num_batches:
                loss = model(X.to(device), y.to(device), None, None, training=False)
                total_loss += loss
        # returning mean
        return total_loss / num_batches
