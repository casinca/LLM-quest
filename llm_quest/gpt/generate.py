import torch


def generate_simple_loop(input, model, max_gen, context_length):
    """
    simplified generate_loop function
    """
    for i in range(max_gen):
        # truncate token ids to compatible context size, shape (b, seq_len) → (b, ctx_len)
        trunc_input = input[:, -context_length:]

        with torch.inference_mode():  # no need for grad as we're generating
            logits = model(trunc_input)
        # taking last vector since goal is "next word" prediction, and getting idx of the highest logit
        logits = logits[:, -1, :]
        tok_id_next = torch.argmax(logits, dim=-1, keepdim=True)  # keepdim to concat (needs same dim)
        input = torch.cat((input, tok_id_next), dim=-1)  # adding predicted token id back to the input for the next loop

    # final "input" is actually initial input+all predicted token ids
    return input


def generate_loop(
    input,
    model,
    max_gen,
    context_length,
    top_k=None,
    temp=0.0,
    eos_id=None,
    device="cuda",
):
    """
    Generates text using a GPT model with optional top-k sampling, temperature scaling, and early stopping.

    Args:
        input (torch.Tensor): Input tensor of token IDs with shape [batch_size, seq_len]
        model (nn.Module): The gpt model used for text generation
        max_gen (int): Maximum number of tokens to generate
        context_length (int): Maximum context length the model can process
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        temp (float, optional): Temperature for softmax sampling:
                                - if >1, increases entropy (randomness)
                                - if <1, decreases entropy (more deterministic)
                                - if 1, untempered distribution
                                - if 0, uses greedy sampling. Defaults to 0.0.
        eos_id (int, optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to None.
        device (str, optional): Device to move the input tensor to. Defaults to "cuda".

    Returns:
        torch.Tensor: Input tensor concatenated with generated token IDs
    """
    input = input.to(device)

    for i in range(max_gen):
        # truncate input to compatible context size, shape (b, ctx_len)
        trunc_input = input[:, -context_length:]

        with torch.inference_mode():  # no need for grads as we're generating
            logits = model(trunc_input)
        # taking last vector since goal is "next word" prediction
        logits = logits[:, -1, :]

        if top_k:
            logits = top_k_sampling(logits, top_k)
        if temp > 0:
            logits = logits / temp
            probas = torch.softmax(logits, dim=-1)
            tok_id_next = torch.multinomial(probas, num_samples=1)  # next tok id is taken from the prob distrib
        else:
            tok_id_next = torch.argmax(logits, dim=-1, keepdim=True)  # keepdim as it is to concat (needs same dim)

        if tok_id_next == eos_id:  # if a EoT is seen stops the generation earlier
            break

        input = torch.cat((input, tok_id_next), dim=-1)  # adding chosen token id back to the input for the next loop

    # final "input" is actually initial input+all predicted words
    return input


def top_k_sampling(logits, k):
    """
    Performs top-k sampling on the input logits by keeping only the k highest probability tokens.

    Args:
        logits (torch.Tensor): Input logits tensor representing token raw probabilities (scores)
        k (int): Number of top tokens to keep

    Returns:
        torch.Tensor: Filtered logits tensor with only the top k values preserved and
                    all others set to negative infinity
    """
    top_k, top_idx = torch.topk(logits, k)
    # initiate a tensor of the size of logits with all values set to -inf
    filt_logits = torch.full_like(logits, -torch.inf)
    # mapping top k values back via their idx, on the given dim (-1), in place
    filt_logits.scatter_(-1, top_idx, top_k)

    return filt_logits


# test code
if __name__ == "__main__":

    import tiktoken

    import config
    from gpt_download import download_and_load_gpt2
    from llm_quest.gpt.gpt_model import GPTModel
    from llm_quest.utils import ids_to_text, load_weights_into_gpt, text_to_ids

    # ---------------------------- PART A ------- Testing simple func generation without pretrained weights

    # print("START")
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)

    # model.eval()
    # model = GPTModel(GPT_CONFIG_124M)

    ## Extracting model parameters for MultiHeadAttention
    # print(sum(p.numel() for p in model.trf_blocks[0].att.parameters()))

    # output = model(batch)
    # print(output.shape)

    # starting_txt = "Hello, i am a model"
    # tokenized_txt = tokenizer.encode(starting_txt)
    ## wrapping in a batch dim as expected input is (b, tok_ids) shape
    # tokenized_tensor = torch.tensor(tokenized_txt).unsqueeze(0)
    # print(tokenized_tensor.shape)

    # output = generate_simple_loop(tokenized_tensor, model, 3, GPT_CONFIG_124M["context_length"])
    # print(output)
    # print(tokenizer.decode(output.squeeze(0).tolist()))

    # ---------------------------- PART B ------- Testing generation with weights trained on "the_verdict"

    # tokenizer = tiktoken.get_encoding("gpt2")

    ## loading our custom pretrained config/model params
    # model = GPTModel(GPT_SMALL_CONFIG)
    # checkpoint = torch.load(config.custom_pretrained_w_gpt2)
    # model.load_state_dict(checkpoint["model_state_dict"])
    #
    # torch.manual_seed(123)
    # model.eval()
    #
    # output2 = generate_loop(
    #    input=text_to_ids("Every effort moves you", tokenizer=tokenizer),
    #    model=model,
    #    max_gen=15,
    #    context_length=GPT_SMALL_CONFIG["context_length"],
    #    top_k=25,
    #    temp=1.4,
    # )
    #
    # print(f"OUTPUT2: {ids_to_text(output2, tokenizer)}")

    # ---------------------------- PART C ------- Testing generation with OpenAI's pretrained weights

    settings, params = download_and_load_gpt2(model_size="124M", models_dir=config.openai_pretrained_w_gpt2)

    tokenizer = tiktoken.get_encoding("gpt2")
    model_settings = config.config_creator("gpt_s")
    torch.manual_seed(123)

    device = "cuda"
    model = GPTModel(model_settings)
    model.eval()

    load_weights_into_gpt(model, params)

    model.to(device)  # we move the model to GPU *after* loading weights

    output3 = generate_loop(
        input=text_to_ids("This is where it", tokenizer=tokenizer).repeat_interleave(3, dim=0),
        model=model,
        max_gen=20,
        context_length=model_settings["context_length"],
        top_k=25,
        temp=1.4,
    )

    for tensor in output3:
        print(ids_to_text(tensor, tokenizer))
