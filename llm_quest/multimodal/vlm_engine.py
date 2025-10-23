import torch
import torch.nn.functional as F


def get_embeddings(text_input, model):
    """
    helper function to return token + positional embeddings for the text input

    Args:
        text_input(torch.tensor): text input ids, shape (batch_size, seq_len)
        model: gpt2 model

    returns:
        (torch.tensor): Combined token + positional embeddings, shape (batch_size, seq_len, emb_dim)
    """
    token_embeddings = model.emb_dict(text_input)
    positions = torch.arange(token_embeddings.shape[1], dtype=torch.long, device=token_embeddings.device)
    position_embeddings = model.pos_emb_dict(positions)

    return token_embeddings + position_embeddings


def vlm_loss(logits, labels, text_attention_mask, num_vision_tokens):
    """
    Classic NTP with a CE loss
    We want to predict the text tokens given the image + text context
    so image tokens/patches+padding are ignored, text_tokens are predicted

    Args:
        logits: (torch.tensor): shape (batch, seq_len, vocab_size)
        labels: (torch.tensor): shape (batch, seq_len)
        text_attention_mask: (torch.tensor): shape (batch, seq_len)
        num_vision_tokens: (int): number of vision tokens in the input
    """

    shifted_logits = logits[:, num_vision_tokens - 1 : -1, :]  # the last vision token predicts the first text token
    labels = labels.masked_fill(text_attention_mask == 0, -100)  # already aligned with shifted_logits

    loss = F.cross_entropy(shifted_logits.flatten(0, 1), labels.flatten(), ignore_index=-100)

    return loss


def vlm_training_loop_simple(
    vit_model,
    vlm_model,
    adapter,
    train_loader,
    optimizer,
    num_epochs,
    device,
    hf_vit_model=True,
    val_loader=None,
    eval_freq=None,
    eval_iter=None,
):
    """
    Training loop for multimodal ViT-GPT2 model.

    Args:
        vit_model (ViTModel): Vision Transformer model
        vlm_model (GPTModel): GPT2 model
        adapter (ViTAdapter): Adapter to connect ViT to GPT2
        train_loader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates
        num_epochs (int): Number of epochs to train for
        device (torch.device): Device to run training on
        hf_vit_model (bool): whether the ViT model is from huggingface or from scrach (different output signature)
        val_loader (DataLoader, optional): DataLoader for validation data. If None, no validation is performed
        eval_freq (int, optional): Number of steps between evaluations. Required if val_loader is provided
        eval_iter (int, optional): Number of batches to use during evaluation. Required if val_loader is provided

    Returns:
        tuple: A tuple containing:
            - vlm_model (GPTModel): The trained GPT model
            - adapter (ViTAdapter): The trained adapter
    """

    # freezing ViT
    vit_model.eval()
    for param in vit_model.parameters():
        param.requires_grad = False

    vlm_model.train()
    adapter.train()

    vit_model.to(device)
    vlm_model.to(device)
    adapter.to(device)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)  # shape (batch, 3, image_size(224), image_size(224))
            input_ids = batch["input_ids"].to(device)  # shape (batch, max_caption_len)
            text_attention_mask = batch["attention_mask"].to(device)  # shape (batch, max_caption_len)

            # Vision
            if not hf_vit_model:
                vit_hidden_states = vit_model(images, output_hidden_states=True)  # shape (b, n_patches+1, vit_h_dim)
            else:
                vit_hidden_states = vit_model(images).last_hidden_state
            # projecting/transforms to the LLM embedding dimension
            vision_embeddings = adapter(vit_hidden_states)  # shape: (b, num_patches+1, llm_d_in)

            # Text
            text_embeddings = get_embeddings(input_ids, vlm_model)  # shape (batch, seq_len, llm_d_in)

            # Early Fusion: concat patches+text embeddings
            batch_size = images.shape[0]
            num_vision_tokens = vision_embeddings.shape[1]  # num_patches+1
            # combine embeddings, shape (batch, num_patches+1 + seq_len, llm_d_in)
            combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)

            # combine attention masks
            # vision patches: always attend (all 1s)
            vision_mask = torch.ones(batch_size, num_vision_tokens, dtype=torch.bool, device=device)
            combined_attention_mask = torch.cat([vision_mask, text_attention_mask], dim=1)  # (b, n_patches+1+seq_len)

            # Forward pass
            logits = vlm_model(  # shape (batch, num_patches+1 + seq_len, vocab_size)
                combined_embeddings,
                attn_mask=combined_attention_mask,
                input_embedded=True,
            )

            loss = vlm_loss(logits, input_ids, text_attention_mask, num_vision_tokens)
            loss.backward()

            total_loss += loss.item()

            # Update weights (vlm model + adapter)
            torch.nn.utils.clip_grad_norm_(list(vlm_model.parameters()) + list(adapter.parameters()), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # --- Evaluation ---
            if val_loader is not None and eval_freq is not None and (step + 1) % eval_freq == 0:
                train_loss, val_loss = vlm_evaluation(
                    train_loader,
                    val_loader,
                    vit_model,
                    adapter,
                    vlm_model,
                    eval_iter,
                    device,
                    hf_vit_model,
                )

                print(
                    f"Epoch: {epoch}, Step: {step+1}",
                    f"Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}",
                    f"Δ: {val_loss - train_loss:.3f} ({((val_loss - train_loss) / train_loss * 100):.2f}%)",
                )

            # regular training progress logging
            if val_loader is None and (step + 1) % eval_freq == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch}, step {step+1}, Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch} completed. Average Loss: {total_loss / len(train_loader):.4f}")

    return vlm_model, adapter


def _calc_loss_batch_vlm(
    images,
    input_ids,
    text_attention_mask,
    vit_model,
    adapter,
    vlm_model,
    device,
    hf_vit_model=True,
):
    """
    Calculates the loss for a single batch of multimodal data (images + captions).
    (Used for evaluation only)

    Args:
        images (torch.Tensor): The input image tensor. shape: (batch, 3, H, W)
        input_ids (torch.Tensor): The tokenized caption input IDs. Shape: (batch, seq_len)
        text_attention_mask (torch.Tensor): The attention mask for captions. Shape: (batch, seq_len)
        vit_model (nn.Module): The Vision Transformer model
        adapter (nn.Module): The adapter connecting ViT to GPT
        vlm_model (nn.Module): The GPT model
        device (torch.device): The device on which the model and tensors are located
        hf_vit_model (bool): Whether the ViT model is from HuggingFace or from scratch

    Returns:
        torch.Tensor: The loss for the batch
    """
    images = images.to(device)
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)

    # Vision
    if not hf_vit_model:
        vit_hidden_states = vit_model(images, output_hidden_states=True)
    else:
        vit_hidden_states = vit_model(images).last_hidden_state
    vision_embeddings = adapter(vit_hidden_states)

    # Text
    text_embeddings = get_embeddings(input_ids, vlm_model)

    # Early Fusion
    batch_size = images.shape[0]
    num_vision_tokens = vision_embeddings.shape[1]
    combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)

    # combine attention masks
    vision_mask = torch.ones(batch_size, num_vision_tokens, dtype=torch.bool, device=device)
    combined_attention_mask = torch.cat([vision_mask, text_attention_mask], dim=1)

    logits = vlm_model(combined_embeddings, attn_mask=combined_attention_mask, input_embedded=True)
    loss = vlm_loss(logits, input_ids, text_attention_mask, num_vision_tokens)

    return loss


def calc_loss_loader_vlm(
    dataloader,
    vit_model,
    adapter,
    vlm_model,
    device,
    num_batches=None,
    hf_vit_model=True,
):
    """
    Calculates the average loss across a custom number of batches for a multimodal dataloader.

    Args:
        dataloader (DataLoader): The DataLoader containing batches of multimodal data (images + captions)
        vit_model (nn.Module): The Vision Transformer model
        adapter (nn.Module): The adapter connecting ViT to GPT
        vlm_model (nn.Module): The GPT model
        device (torch.device): The device on which the model and tensors are located
        num_batches (int, optional): The number of batches to evaluate. If None, evaluates all batches
        hf_vit_model (bool): Whether the ViT model is from HuggingFace or from scratch

    Returns:
        (float): The average loss across the specified number of batches. Returns NaN if dataloader is empty
    """
    total_loss = 0

    # Check for smaller num of batches in order to speed up evaluation
    if len(dataloader) == 0:
        return float("NaN")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, batch in enumerate(dataloader):
        if i < num_batches:
            images = batch["image"]
            input_ids = batch["input_ids"]
            text_attention_mask = batch["attention_mask"]

            loss = _calc_loss_batch_vlm(
                images, input_ids, text_attention_mask, vit_model, adapter, vlm_model, device, hf_vit_model
            )
            total_loss += loss.item()

    # return mean loss
    return total_loss / num_batches


def vlm_evaluation(
    train_loader,
    val_loader,
    vit_model,
    adapter,
    vlm_model,
    eval_iter,
    device,
    hf_vit_model=True,
):
    """
    Evaluates the vlm model's performance on training and validation datasets.

    Args:
        train_loader (DataLoader): DataLoader containing the training data batches
        val_loader (DataLoader): DataLoader containing the validation data batches
        vit_model (nn.Module): The Vision Transformer model
        adapter (nn.Module): The adapter connecting ViT to GPT
        vlm_model (nn.Module): The GPT model to evaluate
        eval_iter (int): Number of batches to use/iterate for evaluation
        device (torch.device): The device to run evaluation on
        hf_vit_model (bool): Whether the ViT model is from HuggingFace or from scratch

    Returns:
        tuple: A tuple containing:
            - train_loss (float): The average loss on the training dataset/num_batches
            - val_loss (float): The average loss on the validation dataset/num_batches
    """
    vit_model.eval()
    adapter.eval()
    vlm_model.eval()

    with torch.inference_mode():
        train_loss = calc_loss_loader_vlm(
            train_loader, vit_model, adapter, vlm_model, device, num_batches=eval_iter, hf_vit_model=hf_vit_model
        )
        val_loss = calc_loss_loader_vlm(
            val_loader, vit_model, adapter, vlm_model, device, num_batches=eval_iter, hf_vit_model=hf_vit_model
        )

    # trainable models back to train mode
    adapter.train()
    vlm_model.train()

    return train_loss, val_loss
