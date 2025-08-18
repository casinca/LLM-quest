# A reimplementation of the Number Token Loss - Wasserstein Distance variant (NTL-WAS) from the paper:
#
# Regress, Don’t Guess – A Regression-like Loss on Number Tokens for Language Models
# https://arxiv.org/abs/2411.02083
#
# Code repo: https://github.com/ai4sd/number-token-loss (old: https://github.com/tum-ai/number-token-loss)
#
# NTL-WAS is more interesting, as NTL with MSE can be suboptimal, with different combination of dot products potentially
# giving a correct answer despite being wrong. That is because of the weighted average matching the label:
# Ex: label=4 and pred:probs are 3=0.5 and 5=0.5, res = 3*0.5+5*0.5 = 4.
# L_mse = (correct value - predicted value)² = (4-4)² = 0 → wrong answer and not penalized.
import torch
from torch.nn import functional as F


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
        self.num_tokens_mask = ~torch.isnan(self.num_vocab)

    def _build_num_vocab_tensor(self, tokenizer):
        """
        returns:
            tensor of shape (vocab_size,) where digits are mapped to their float value and other tokens are mapped to
            NaN.
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

    # pretty much following the code
    def _calc_ntl_was(self, logits, labels, importance_mask=None, ignore_index=-100):
        """
        NumTokenLoss Wasserstein Distance variant (NTL-WAS)
        """

        # setting padding/ignored tokens' value to NaN, by switching index to 0 (which has a value of NaN in num_vocab)
        labels = labels.masked_fill(labels == ignore_index, 0)
        # creating mask for number tokens in the labels
        labels_values = self.num_vocab[labels]
        valid_labels_mask = ~torch.isnan(labels_values)

        # mask to weight the loss of number tokens based on their importance
        if importance_mask is not None:
            labels_imp_mask = importance_mask[valid_labels_mask]

        # if there's no valid labels (number tokens) or all tokens are masked:
        #  we return the loss as 0 (ie no loss added to the main one)
        if not valid_labels_mask.any() or (importance_mask is not None and not importance_mask.any()):
            return 0.0

        # --- Loss calculation ---
        # get probs of number tokens predictions
        number_logits = logits[:, :, self.num_tokens_mask]
        number_probs = F.softmax(number_logits, dim=-1)

        # they use absolute difference between the true numbers and all possible number values, weighted by their probs.
        # They leverage the fact that labels are one-hot encoded. No need to compute the difference of CDFs.
        costs_to_labels = torch.abs(
            labels_values[valid_labels_mask].unsqueeze(-1) - self.num_vocab[self.num_tokens_mask]
        )
        loss = (costs_to_labels * number_probs[valid_labels_mask]).sum(-1)

        if importance_mask is not None:
            # normalized by the number of tokens that have a non-zero weight
            loss = (loss * labels_imp_mask).sum() / labels_imp_mask.count_nonzero()

        return loss.mean()  # (for conciseness, if importance mask: mean of a scalar is just the scalar itself)

    def __call__(self, logits, labels, importance_mask=None, ignore_index=-100):
        return self._calc_ntl_was(logits, labels, importance_mask, ignore_index)
