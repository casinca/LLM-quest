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


def multimodal_loss(logits, labels, text_attention_mask, num_vision_tokens):
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


def multimodal_training_loop_simple(
    vit_model,
    multimodal_model,
    adapter,
    train_loader,
    optimizer,
    num_epochs,
    device,
    hf_vit_model=True,
):
    """
    Training loop for multimodal ViT-GPT2 model.

    Args:
        vit_model (ViTModel): Vision Transformer model
        multimodal_model (GPTModel): GPT2 model
        adapter (ViTAdapter): Adapter to connect ViT to GPT2
        train_loader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates
        num_epochs (int): Number of epochs to train for
        device (torch.device): Device to run training on
        hf_vit_model (bool): whether the ViT model is from huggingface or from scrach (different output signature)
    """

    # freezing ViT
    vit_model.eval()
    for param in vit_model.parameters():
        param.requires_grad = False

    multimodal_model.train()
    adapter.train()

    vit_model.to(device)
    multimodal_model.to(device)
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
            text_embeddings = get_embeddings(input_ids, multimodal_model)  # shape (batch, seq_len, llm_d_in)

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
            logits = multimodal_model(  # shape (batch, num_patches+1 + seq_len, vocab_size)
                combined_embeddings,
                attn_mask=combined_attention_mask,
                input_embedded=True,
            )

            loss = multimodal_loss(logits, input_ids, text_attention_mask, num_vision_tokens)
            loss.backward()

            total_loss += loss.item()

            # Update weights (multimodal model + adapter)
            torch.nn.utils.clip_grad_norm_(
                list(multimodal_model.parameters()) + list(adapter.parameters()), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch}, step {step+1}, Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch} completed. Average Loss: {total_loss / len(train_loader):.4f}")

    return multimodal_model, adapter
