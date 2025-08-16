import torch


class NumTokenLoss:
    def __init__(self, tokenizer, ntl_coeff=0.3, device=None, multi_digits=False):
        """
        tokenizer (Tokenizer): tokenizer to use: tiktoken or transformers
        ntl_coeff (float): weight of the NumTokenLoss (default: 0.3)
        device (str): device to use for the NumTokenLoss
        multi_digits (bool): whether to allow token to be a multiple digit number (default: False)
        """
        self.ntl_coeff = ntl_coeff
        self.device = device
        self.multi_digits = multi_digits

        self.vocab_map = self.get_num_vocab_map(tokenizer)

    def get_num_vocab_map(self, tokenizer):
        # --- retrieve vocab from the tokenizer ---
        # transformers (default)
        try:
            vocab = tokenizer.get_vocab()

        # tiktoken alt
        except AttributeError:
            try:
                vocab = {
                    tokenizer.decode_single_token_bytes(token_id).decode("utf-8", errors="strict"): token_id
                    for token_id in range(tokenizer.n_vocab)
                }

            # if both fail
            except Exception as e:
                raise RuntimeError(f"Could not retrieve or construct vocabulary from the tokenizer. Error: {e}")

        vocab_map = torch.full((len(vocab),), float("nan"))

        # --- build vocab mapping from the vocab dictionary ---
        for string, token_id in vocab.items():
            stripped_token = string.strip()

            if not stripped_token.isdigit() and stripped_token != "-1":
                continue

            try:
                token_value = float(stripped_token)

                if self.multi_digits:  # experimental: multi-digit tokens (ie "123")
                    vocab_map[token_id] = token_value
                else:  # Follow the paper: only -1 to 9 (single digit)
                    if -1 <= token_value <= 9 and len(stripped_token) == 1:
                        vocab_map[token_id] = token_value

            except ValueError:
                print(f"!!! Parsing failed: Could not convert token {string} to float. Skipping...")
                continue

        return vocab_map

    def calc_ntl_was(self):
        """
        NumTokenLoss Wasserstein Distance variant (NTL-WAS)
        """
        pass

    def __call__(self):
        pass
