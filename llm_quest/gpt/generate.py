import torch


def generate_simple_loop(input_tensor, model, max_gen, context_length):
    """
    simplified generate_loop function
    """
    for i in range(max_gen):
        # truncate token ids to compatible context size, shape (b, seq_len) → (b, ctx_len)
        trunc_input = input_tensor[:, -context_length:]

        with torch.inference_mode():  # no need for grad as we're generating
            logits = model(trunc_input)
        # taking last vector since goal is "next word" prediction, and getting idx of the highest logit
        logits = logits[:, -1, :]
        tok_id_next = torch.argmax(logits, dim=-1, keepdim=True)  # keepdim to concat (needs same dim)
        input_tensor = torch.cat(
            (input_tensor, tok_id_next), dim=-1
        )  # adding predicted token id back to the input for the next loop

    # final "input" is actually initial input+all predicted token ids
    return input_tensor


def generate_loop(
    input_tensor,
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
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs.
        model (nn.Module): The GPT model used for text generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
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
    input_tensor = input_tensor.to(device)

    for i in range(max_gen):
        # truncate input to compatible context size, shape (b, ctx_len)
        trunc_input = input_tensor[:, -context_length:]

        with torch.inference_mode():  # no need for grads as we're generating
            logits = model(trunc_input)[:, -1, :]  # taking last vector (next word prediction)

        if top_k:
            logits = top_k_sampling(logits, top_k)
        if temp > 0:
            tok_id_next = sample_with_temperature(logits, temp)  # next tok id is taken from the prob distrib
        else:
            tok_id_next = torch.argmax(logits, dim=-1, keepdim=True)  # keepdim as it is to concat (needs same dim)

        if tok_id_next == eos_id:  # if a EoT is seen stops the generation earlier
            break

        input_tensor = torch.cat(
            (input_tensor, tok_id_next), dim=-1
        )  # adding chosen token id back to the input for the next loop

    # final "input" is actually initial input+all predicted words
    return input_tensor


# It is a necessary more robust generate_loop() function for batching prompts in the case of RLHF/RLVR:
# NOTE: this is still a bit dirty and done to make proper right padding works, it won't work with left padding atm.
# This is also to avoid having to implement KVCache solely for a single function.
#
# dynamic attention_mask: Avoid padding tokens from shorter prompts while generating
# dynamic generation:
#   - only generating for unfinished prompts (more efficient vs continue useless generations after EoS)
#   - early finished generations are out of the loop and padded with eos_id
# retrieving the last real token's prediction for the first step in the right padding case
def generate_batched_loop(
    input_tensor,
    model,
    max_gen,
    context_length,
    top_k=None,
    temp=0.0,
    eos_id=50256,
    device="cuda",
    attention_mask=None,
    last_real=None,
):
    """
    Generates text from batched prompts, handling dynamic attention masks and early stopping for individual sequences.
    We retrieve the last real token's prediction for the first step for right padding case.

    Args:
        input_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the initial prompt token
        IDs.
        model (nn.Module): The model used for generation.
        max_gen (int): The maximum number of new tokens to generate for each sequence.
        context_length (int): The maximum sequence length (context window) the model can handle.
        top_k (int, optional): If specified, limits sampling to top k most likely tokens. Defaults to None.
        temp (float, optional): Sampling temperature. A higher value makes the output more random.
                                Defaults to 0.0 (greedy sampling).
        eos_id (int, optional): Token ID that signals end of text. Generation stops early if encountered.
                                Defaults to 50256 (GPT-2 EOS token).
        device (str, optional): The device to perform computations on (e.g., "cuda" or "cpu"). Defaults to "cuda".
        attention_mask (torch.Tensor, optional): A boolean tensor of shape (batch_size, sequence_length) indicating
                                                which tokens are real (True) and which are padding (False).
                                                Used during attention calculation. Defaults to None.
        last_real (torch.Tensor, optional): A tensor of shape (batch_size,) indicating the index of the last real token
                                            in each prompt of `input_tensor`. Used in the first generation step to
                                            correctly extract logits for right-padded inputs. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, prompt_length + generated_length) containing the original prompts
                        concatenated with the generated token IDs. Sequences that finished early will be padded with
                        `eos_id` up to `max_gen` length.
    """
    input_tensor = input_tensor.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if last_real is not None:
        last_real = last_real.to(device)

    batch_size = input_tensor.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # tracking if a generation is finished

    for step in range(max_gen):
        if finished.all():  # early exit
            break

        # we build the input only for unfinished rows/sequences
        # for simplicity, calling "N_active" the number of True values in the mask/batch.
        unfinished = ~finished  # bool mask of unfinished generations (batch_size,)
        trunc_input = input_tensor[unfinished, -context_length:]

        curr_mask = None
        if attention_mask is not None:
            curr_mask = attention_mask[unfinished, -context_length:]

        with torch.inference_mode():
            logits = model(trunc_input, attn_mask=curr_mask)  # (N_active, seq_len, v)

        if step == 0 and last_real is not None:
            seq_pos = torch.arange(batch_size, device=device)  # N_active=batch size for the first step
            logits = logits[seq_pos, last_real[unfinished], :]
        else:
            logits = logits[:, -1, :]  # (N_active, v)

        # sample for the N_active/unfinished rows
        if top_k:
            logits = top_k_sampling(logits, top_k)

        if temp > 0:
            next_toks = sample_with_temperature(logits, temp)  # (N_active, 1)
        else:
            next_toks = logits.argmax(dim=-1, keepdim=True)  # (N_active, 1)

        # create a tensor full of eos_id, insert sampled tokens back into the batch only for unfinished generations
        full_next_toks = torch.full((batch_size, 1), eos_id, device=device, dtype=torch.long)
        full_next_toks[unfinished] = next_toks

        # append to running tensors
        input_tensor = torch.cat([input_tensor, full_next_toks], dim=-1)
        if attention_mask is not None:
            new_mask = torch.ones_like(full_next_toks, dtype=torch.bool)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        # update finished flag
        finished |= full_next_toks.squeeze(1) == eos_id

    return input_tensor


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


def sample_with_temperature(logits, temp):
    """
    Performs temperature scaling on the input logits and samples from the resulting distribution

    Args:
        logits (torch.Tensor): Input logits tensor representing token raw probabilities (scores), shape (b, v)
        temp (float): Temperature for softmax sampling:
                        - if >1, increases entropy (randomness)
                        - if <1, decreases entropy (more deterministic)
                        - if 1, untempered distribution

    Returns:
        torch.Tensor: Sampled token IDs from the distribution, shape (b, 1)
    """
    probs = torch.softmax(logits / temp, dim=-1)
    tok_id_next = torch.multinomial(probs, num_samples=1)
    return tok_id_next


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
        input_tensor=text_to_ids("This is where it", tokenizer=tokenizer).repeat_interleave(3, dim=0),
        model=model,
        max_gen=20,
        context_length=model_settings["context_length"],
        top_k=25,
        temp=1.4,
    )

    for tensor in output3:
        print(ids_to_text(tensor, tokenizer))
