import torch


class NumTokenLoss:
    def __init__(self, tokenizer, ntl_coeff=0.3, device=None, multi_digits=False):
        """
        tokenizer (Tokenizer): transformers tokenizer to use
        ntl_coeff (float): weight of the NumTokenLoss (default: 0.3)
        device (str): device to use for the NumTokenLoss
        multi_digits (bool): whether to allow token to be a multiple digit number (ie "123")
        """
        self.ntl_coeff = ntl_coeff
        self.device = device
        self.multi_digits = multi_digits

        self.vocab_map = self.get_num_vocab_map(tokenizer)

    def get_num_vocab_map(self, tokenizer):
        # retrieve vocab from the tokenizer
        vocab = tokenizer.get_vocab()
        vocab_map = torch.full((len(vocab),), float("nan"))

        # build vocab mapping from the vocab dictionary
        for string, token_id in vocab.items():
            stripped_token = string.strip()

            try:
                token_value = float(stripped_token)
                single_digit = -1 <= token_value <= 9 and len(stripped_token) == 1

                if self.multi_digits or single_digit:
                    vocab_map[token_id] = token_value

            except ValueError:
                # print(f"!!! Parsing failed: Could not convert token {string} to float. Skipping...")
                continue

        return vocab_map

    def calc_ntl_was(self):
        """
        NumTokenLoss Wasserstein Distance variant (NTL-WAS)
        """
        pass

    def __call__(self):
        pass
