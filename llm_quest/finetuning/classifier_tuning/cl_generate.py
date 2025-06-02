import tiktoken
import torch
import torch.nn as nn

import config
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.utils import text_to_ids

# NOTE:  Same as for training.
# The way classify_text was implemented here and llm-from-scratch, accepts a single sentence at a time.
# there's no need to pad a single sentence, but truncation is still needed if > ctx_len.
#
# Here, if we keep the code as it is, we'll still get the problem of retrieving the last token's logits, since we
# are padding we will retrieve a padded token and not a valid one, unless seq_len=max_length.
# Even though it works since padded tokens gets context propagated, it's not the best way to do it.


def classify_text(text, model, device, max_length=None, pad_token=50256):
    """
    A classification function inspired by our previous generate_loop() and SpamDataset class
    """
    model.eval()
    input = text_to_ids(text, tokenizer=tokenizer)
    # calc max length and input length
    ctx_len = model.pos_emb_dict.weight.shape[0]  # shape (ctx_len, emb_dim)
    max_length = min(max_length, ctx_len) if max_length else ctx_len
    input_len = min(input.shape[1], max_length)

    # truncating and padding
    # creating a tensor full of pads token of size max_length
    pads = torch.full((input.shape[0], max_length), pad_token)
    # replace all pads token below input_len by the corresponding inputs
    pads[:, :input_len] = input[:, :input_len]

    # This is only as a trick to retrieve the last token's logits, We don't need attn_mask for generation
    attn_mask = torch.zeros_like(pads, dtype=torch.bool)
    attn_mask[:, :input_len] = 1

    with torch.no_grad():
        logits = model(pads, only_last_token=True, attn_mask=attn_mask)
        pred = torch.argmax(logits, dim=-1)

        # (optional) checking confidence
        print(f"{torch.max(torch.softmax(logits, dim=-1)).item() * 100:.2f}%")

    return "SPAM" if pred == 1 else "Not SPAM"


if __name__ == "__main__":

    tokenizer = tiktoken.get_encoding("gpt2")

    model_config = config.config_creator("gpt_s")
    model_config["drop_rate"] = 0.0
    model = GPTModel(model_config)

    # changing the arch from the classic GPT2 to the classification output head, just like in cl_training.py
    # alternatively, could have saved the whole model arch during cl_training (and not just params) instead.
    num_classes = 2
    model.out = nn.Linear(model_config["emb_dim"], num_classes)

    # loading our finetuned config/model params
    ft_checkpoint = torch.load(config.ft_classifier_w_gpt2, weights_only=False)
    model.load_state_dict(ft_checkpoint["model_state_dict"])

    text_spam = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."

    output = classify_text(text_spam, model, device="cuda", max_length=35)

    print(output)
