import torch
from torchvision.transforms import functional as F

from llm_quest.gpt.generate import sampling


def vlm_generate_loop(
    image,
    vit_model,
    adapter,
    vlm_model,
    tokenizer,
    max_gen=70,
    context_length=300,
    top_k=None,
    top_p=0.9,
    temp=0.8,
    device="cuda",
    hf_vit_model=True,
    image_size=224,
):
    """
    Generates image captions/description using a trained VLM model (using ViT + Adapter + GPT2).
    The setup is similar to the classic generate_loop() function with some differences:
    - We preprocess the image just like for training, to get it enriched/aligned and ready for the VLM
    - We are working in embedding space (not token IDs)

    Args:
        image: PIL Image
        vit_model: Vision Transformer for getting image embeddings hidden states
        adapter: ViT to GPT embedding adapter
        vlm_model: VLM model for text generation
        tokenizer: Tokenizer for text generation
        max_gen(int): Maximum tokens to generate
        context_length(int): Context window of the VLM model
        top_k, top_p, temp: classic sampling parameters
        device: Device to run inference on
        hf_vit_model (bool): Whether the ViT model is from HuggingFace or from scratch
        image_size (int): Size of the image to resize to (default: 224)

    Returns:
        caption (str): Generated caption/description string based on the image fed to the VLM.
    """

    def _preprocess_image(image, image_size=image_size):
        """
        Standalone helper function to preprocess image for ViT input (same as in MultimodalDataset).
        """
        img = F.resize(image, [image_size, image_size])
        tensor = F.to_tensor(img)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor

    vit_model.eval().to(device)
    adapter.eval().to(device)
    vlm_model.eval().to(device)

    with torch.inference_mode():
        # Preprocess image
        image_tensor = _preprocess_image(image).unsqueeze(0).to(device)  # shape (batch(1), ch(3), img_size(224), (224)
        if not hf_vit_model:
            vit_hidden_states = vit_model(image_tensor, output_hidden_states=True)  # shape (b, n_patches+1, vit_h_dim)
        else:
            vit_hidden_states = vit_model(image_tensor).last_hidden_state
        vision_embeddings = adapter(vit_hidden_states)

        # We trained the VLM without any special token between end of vision and start of text tokens.
        # So we expect the VLM to start generating text tokens right after the vision tokens.
        input_sequence = vision_embeddings  # (1, num_patches+1 (197), emb_dim)

        generated_token_ids = []

        for i in range(max_gen):
            # Truncate if needed, as the sequence grows: 197 (vision) + generated tokens
            trunc_input = input_sequence[:, -context_length:]

            # using inputs_embedded because we're working in embedding space
            logits = vlm_model(trunc_input, input_embedded=True)
            logits = logits[:, -1, :]  # shape (1, vocab_size)

            # Sample next token
            next_token_id = sampling(logits, top_k, top_p, temp)
            next_token_id_scalar = next_token_id.item()
            generated_token_ids.append(next_token_id_scalar)

            if next_token_id_scalar == tokenizer.eos_token_id:
                break

            # convert sampled token ID to embedding for the next iteration
            # (get the embedding from the VLM's token embedding layer)
            next_token_embedding = vlm_model.emb_dict(next_token_id)  # shape (1, 1, emb_dim)

            input_sequence = torch.cat((input_sequence, next_token_embedding), dim=1)

        caption = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, ViTModel

    import config
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.vision_transformer.vit_engine import ViTAdapter

    torch.manual_seed(123)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test image
    ds_test = load_dataset("jxie/flickr8k", split="test")
    image = ds_test[0]["image"]
    image.show()  # 2 brownish dogs playing together in the snow

    # Initialize models and adapter
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    vit_model = vit_model.eval()

    gpt_config = config.config_creator("gpt_s")
    vlm_model = GPTModel(gpt_config)
    vlm_model.load_state_dict(torch.load(config.vlm_gpt, map_location="cpu"))

    adapter = ViTAdapter(
        vit_d_out=vit_model.config.hidden_size,  # 768
        llm_d_in=gpt_config["emb_dim"],
        adapter_type="ffn",
    )
    adapter.load_state_dict(torch.load(config.vlm_adapter, map_location="cpu"))

    # Generate caption
    caption = vlm_generate_loop(
        image=image,
        vit_model=vit_model,
        adapter=adapter,
        vlm_model=vlm_model,
        tokenizer=tokenizer,
        max_gen=60,
        context_length=gpt_config["context_length"],
        temp=1.0,
        top_p=0.95,
    )

    print(caption)
