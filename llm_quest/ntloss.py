# A reimplementation of the Number Token Loss - Wasserstein Distance variant (NTL-WAS) from the paper:
#
# Regress, Don’t Guess – A Regression-like Loss on Number Tokens for Language Models
# https://arxiv.org/abs/2411.02083
#
# Code repo: https://github.com/tum-ai/number-token-loss/
#
# NTL-WAS is more interesting, as NTL with MSE can be suboptimal (with different combination of dot products potentially
# giving a correct answer despite being wrong because of the weighted average matching the label):
# Ex: label=4 and pred:probs are 3=0.5 and 5=0.5, res = 3*0.5+5*0.5 = 4.
# L_mse = (correct value - predicted value)² = (4-4)² = 0 → wrong answer and not penalized.
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

        self.num_vocab = self._build_num_vocab_tensor(tokenizer)
        self.is_number_token = ~torch.isnan(self.num_vocab)

    def _build_num_vocab_tensor(self, tokenizer):
        """
        returns:
            tensor of shape (vocab_size,) where digits are mapped to their float value and other tokens are mapped to
            nan.
        """

        # retrieve vocab from the tokenizer
        vocab = tokenizer.get_vocab()
        num_vocab = torch.full((len(vocab),), float("nan"))

        # build vocab mapping from the vocab dictionary
        for string, token_id in vocab.items():
            stripped_token = string.strip()

            try:
                token_value = float(stripped_token)
                single_digit = -1 <= token_value <= 9 and len(stripped_token) == 1  # paper default: single digit only

                if self.multi_digits or single_digit:
                    num_vocab[token_id] = token_value

            except ValueError:
                # print(f"!!! Parsing failed: Could not convert token {string} to float. Skipping...")
                continue

        return num_vocab

    def calc_ntl_was(self):
        """
        NumTokenLoss Wasserstein Distance variant (NTL-WAS)
        """
        pass

    def __call__(self):
        pass
