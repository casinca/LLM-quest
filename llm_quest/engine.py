import math
import time

import torch
from torch.amp import GradScaler


def _calc_loss_batch(X, y, model, device, attn_mask=None, classification=False):
    """
    (Used for evaluation only)
    Calculates the CE loss for a batch of data.
    see global_loss() for training loss calc

    Args:
        X (torch.Tensor): The input tensor for the batch.
        y (torch.Tensor): The target tensor for the batch.
        model (nn.Module): The model used to make predictions.
        device (torch.device): The device on which the model and tensors are located.
        attn_mask (torch.Tensor): The attention mask tensor for the batch.
        classification (boolean): Tweak to adapt the loss calculation for classification finetuning
        (ie, loss calc on last pred) or not

    Note: for ex, pretraining (next word pred) logits will have to be flattened 3D → 2D:
        (b,s, vocab) → (b*s,vocab) and targets too 2D→1D: (b,s) → (b*s) for the CE loss proper format

    Returns:
        torch.Tensor: The calculated loss for the batch.
    """

    # putting on dedicated device
    X = X.to(device)
    y = y.to(device)

    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # in the case of classification finetuning, we're only interested in minimizing the loss based on the last tokens'
    # logits. Slicing will naturally give us a 2D shape that matches nn.F.cross_entropy() requirement.
    # logits will have a 2D shape: (b, num of classes) and targets 1D shape of true classes, thus no need to flatten.
    if classification:
        logits = model(X, only_last_token=True, attn_mask=attn_mask)
        loss = torch.nn.functional.cross_entropy(logits, y)
    # but in classic causal NTP, logits have a shape (b,s,v) and targets (b, s)
    # thus we need to flatten logits to (b*s, v) and targets (b*s) for the correct format
    else:
        logits = model(X, attn_mask=attn_mask)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

    return loss


def global_loss(logits, y, model=None, classification=False):
    """
    Quick wrapper of the CE loss for a batch of data. Based on _calc_loss_batch()

    Used for training only, to calculate the global loss of possible multiple loss terms (eg, MoE).
    _calc_loss_batch() is used for eval to calc the main loss only: CE
    """
    if classification:
        loss = torch.nn.functional.cross_entropy(logits, y)
    else:
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

    # Add MoE auxiliary losses
    # we are not averaging, but adding the cumulated aux losses of all MoE layers
    # TODO might want to also add classic aux loss for DeepSeek and rethink the logic handling for training/inf
    moe_loss = 0.0
    for module in model.modules():
        if hasattr(module, "ffn"):
            ffn = module.ffn
            if hasattr(ffn, "moe_loss"):
                moe_loss += ffn.moe_loss

    return loss + moe_loss


def calc_loss_loader(dataloader, model, device, num_batches=None, classification=False):
    """
    Calculates the average loss across a customized number of batches from a dataloader.
    (loss of each batch is cumulated while iterating through the dataloader and divided by specified num of batches)

    Args:
        dataloader (DataLoader): The DataLoader containing the batches of data.
        model (nn.Module): The model used to make predictions.
        device (torch.device): The device on which the model and tensors are located.
        num_batches (int, optional): The number of batches to evaluate (to speed up). If None, evaluates all batches.
        attn_mask (torch.Tensor, optional): The attention mask tensor for the batch.
        classification (boolean): Tweak to adapt the loss calculation for finetuning (focusing on last pred)

    Returns:
        float: The average loss across the specified number of batches. Returns NaN if dataloader is empty.
    """
    total_loss = 0
    # checks for smaller num of batches in order to speed up evaluation
    if len(dataloader) == 0:
        return float("NaN")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, batch in enumerate(dataloader):
        if i < num_batches:
            if len(batch) == 3:
                X, y, attn_mask = batch
                loss = _calc_loss_batch(X, y, model, device, attn_mask, classification)
            else:
                X, y = batch
                loss = _calc_loss_batch(X, y, model, device, classification=classification)

            total_loss += loss

    # returning mean
    return total_loss / num_batches


def training_eval_loop_simple(
    train_loader,
    val_loader,
    model,
    optimizer,
    num_epoch,
    eval_freq,
    eval_iter,
    device,
    accumulation_steps=1,
):
    """
    A simple training and evaluation loop for a model.

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
    # keeping track of metrics for plotting
    train_losses, val_losses = [], []

    for epoch in range(1, num_epoch + 1):
        model.train()

        for input_batch, targets in train_loader:
            step += 1

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            logits = model(input_batch)
            loss = global_loss(logits, targets, model=model)

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # eval
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )


# @rasbt's function adapted for benchmarking training speed: avg tok/s and memory usage
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
        for input_batch, targets in train_loader:
            step += 1

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            logits = model(input_batch)

            optimizer.zero_grad()
            loss = global_loss(logits, targets, model=model)
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
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device)
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


def training_eval_loop(
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
    """A more robust training+eval loop with learning rate scheduler: warmup, decay (cosine), gradient clipping and
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
    if warmup_percent and warmup_steps:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # Keep a record of metrics for plotting.
    train_losses, val_losses = [], []

    # Initialize gradient scaler for AMP
    scaler = GradScaler("cuda", enabled=use_amp)

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
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(input_batch)
                loss = global_loss(logits, targets, model=model)

            optimizer.zero_grad()

            # mixed precision backward pass and optimizer step
            if use_amp:
                scaler.scale(loss).backward()
                # gradient clipping
                if step >= warmup_steps:
                    scaler.unscale_(optimizer)  # unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                scaler.step(optimizer)
                scaler.update()

            # standard backward pass,
            else:
                loss.backward()
                # gradient clipping at a max norm of 1 (after warmup)
                if step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()

            # eval (AMP disabled for evaluation with torch no_grad in evaluate())
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )

    return train_losses, val_losses


# While the evaluate function, at first glance, seems redundant (recalculating train loss) it is necessary in the case
# we want to evaluate for a specific number of batches (eval_iter) and not necessarily the loss up to x number
# of steps (in this case it would have indeed be better to accumulate the loss in the training loop and only evalulate()
# for the validation set)
def evaluate(train_loader, val_loader, model, eval_iter, device, classification=False):
    """
    Evaluates the model's performance on training and validation datasets.

    Args:
        train_loader (DataLoader): DataLoader containing the training data batches.
        val_loader (DataLoader): DataLoader containing the validation data batches.
        model (nn.Module): The model to evaluate.
        eval_iter (int): Number of batches to use/iterate for evaluation.
        device (torch.device): The device to run evaluation on.
        classification(boolean): boolean value to tweak correct loss func shape for next word pred or classification

    Returns:
        tuple: A tuple containing:
            - train_loss (float): The average loss on the training dataset
            - val_loss (float): The average loss on the validation dataset
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, classification=classification)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, classification=classification)
    model.train()

    return train_loss, val_loss


# deprecated in favor of training_eval_loop func with AMP choice
def training_eval_loop_deprecated(
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
    """A more robust training+eval loop with learning rate warmup, cosine decay and gradient clipping.

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
    """
    step = -1
    total_steps = len(train_loader) * num_epoch
    warmup_steps = int(warmup_percent * total_steps)
    if warmup_percent:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # keeping record of metrics for plotting
    train_losses, val_losses = [], []

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
                if not param_group.get("custom_lr", False):  # only adjust lr for non-custom groups
                    param_group["lr"] = lr

            input_batch = input_batch.to(device)
            targets = targets.to(device)

            logits = model(input_batch)

            optimizer.zero_grad()
            loss = global_loss(logits, targets, model=model)
            loss.backward()

            # gradient clipping at a max norm of 1 (after warmup)
            if step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            if step % eval_freq == 0:
                # eval
                train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                print(
                    f"Epoch: {epoch}, Step: {step}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                )

    return train_losses, val_losses


def profile_training_eval_loop(
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
    # Profiler parameters
    profile_dir="./torch_profile_logs",
    wait=1,
    warmup=1,
    active=3,
    repeat=1,
    record_shapes=True,
    profile_memory=True,
):
    """
    Runs the training_eval_loop with PyTorch Profiler to identify bottlenecks.

    Args:
        # Original training_eval_loop parameters
        ...

        # Profiler parameters
        profile_dir (str): Directory to save profiling results
        wait (int): Number of steps to wait before starting profiling
        warmup (int): Number of steps for warmup
        active (int): Number of steps to actively profile
        repeat (int): Number of repeats of the cycle
        record_shapes (bool): Record tensor shapes
        profile_memory (bool): Profile memory usage

    Returns:
        Tuple containing train_losses and val_losses
    """
    import os

    from torch.profiler import ProfilerActivity, profile

    # Create directory if it doesn't exist
    os.makedirs(profile_dir, exist_ok=True)

    # Define activities to profile
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Define profiler schedule
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    # Keep track of step for profiler
    step_counter = 0

    # Initialize metrics
    train_losses, val_losses = [], []
    step = -1
    total_steps = len(train_loader) * num_epoch
    warmup_steps = int(warmup_percent * total_steps)
    if warmup_percent and warmup_steps:
        lr_increment = (peak_lr - init_lr) / warmup_steps

    # Initialize gradient scaler for AMP
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Create profiler
    with profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:

        for epoch in range(1, num_epoch + 1):
            model.train()

            for input_batch, targets in train_loader:
                step += 1

                # Learning rate scheduler logic
                if step < warmup_steps:
                    lr = init_lr + step * lr_increment
                else:
                    decay_steps = total_steps - warmup_steps
                    curr_step = step - warmup_steps
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * curr_step / decay_steps))
                    lr = min_lr + (peak_lr - min_lr) * cosine_decay

                # Update lr
                for param_group in optimizer.param_groups:
                    if not param_group.get("custom_lr", False):
                        param_group["lr"] = lr

                input_batch = input_batch.to(device)
                targets = targets.to(device)

                # Forward pass with autocast
                with torch.autocast("cuda", enabled=use_amp):
                    logits = model(input_batch)
                    loss = global_loss(logits, targets, model=model)

                optimizer.zero_grad()

                # Mixed precision backward and optimizer step
                if use_amp:
                    scaler.scale(loss).backward()
                    if step >= warmup_steps:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if step >= warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()

                # Evaluation
                if step % eval_freq == 0:
                    train_loss, val_loss = evaluate(train_loader, val_loader, model, eval_iter, device)
                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss.item())

                    print(
                        f"Epoch: {epoch}, Step: {step}",
                        f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                    )

                # Step the profiler
                prof.step()

                # Optional: break early for profiling purposes
                if step_counter >= wait + warmup + active:
                    break

                step_counter += 1

            # Optional: break early for profiling
            if step_counter >= wait + warmup + active:
                break

    print(f"Profiling complete. Logs saved to {profile_dir}")
    print("To view results, run: tensorboard --logdir={profile_dir}")

    return train_losses, val_losses
