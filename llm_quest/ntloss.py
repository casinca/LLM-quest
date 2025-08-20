# A simpler reimplementation of the Number Token Loss - Wasserstein Distance variant (NTL-WAS) from the ICML 2025 paper:
#
# Regress, Don’t Guess – A Regression-like Loss on Number Tokens for Language Models
# https://arxiv.org/abs/2411.02083
#
# Code repo: https://github.com/ai4sd/number-token-loss (old: https://github.com/tum-ai/number-token-loss)
#
# NTL-WAS is more interesting, as NTL MSE can be suboptimal. That is because different combination of dot products
# can potentially give a correct answer despite being wrong. Consequence of the weighted average matching the label.
# Ex: label=4 and num_pred:prob are 3=0.5 and 5=0.5 → res = 3*0.5 + 5*0.5 = 4 (predicted 3 and 5, avg = 4)
# L_mse = (correct value - predicted value)² = (4-4)² = 0 → wrong prediction and not penalized.
import torch
from torch.nn import functional as F


class NumTokenLoss:
    """
    NumTokenLoss Wasserstein Distance variant (NTL-WAS):

    CE loss penalizes all incorrect predictions equally, ignoring potential numerical proximity. If the label is 4 and
    the model predicts 3 or 199, will yield the same loss. 
    NTL aims to give some credit to the model for having a prediction close to the label.
    """
    def __init__(self, tokenizer, device="cuda", multi_digits=False):
        """
        tokenizer (Tokenizer): transformers tokenizer to use
        device (str): device to use for the NumTokenLoss
        multi_digits (bool): whether to allow token to be a multiple digit number (ie "123")
        """
        self.multi_digits = multi_digits
        self.num_nan_vocab = self._get_num_nan_vocab(tokenizer).to(device)
        self.num_tokens_mask = ~torch.isnan(self.num_nan_vocab)
        self.num_only_values = self.num_nan_vocab[self.num_tokens_mask]

        self.distance_matrix = self._get_cached_distance_matrix()

    def _get_cached_distance_matrix(self):
        """
        For small 0-9 digits case: we cache the distance matrix between all number tokens and the label numbers.
        We end up with a matrix where distance_matrix[i][j] = |token_value_i - token_value_j|

            0   1  2  3  4  5  6  7  8  9   PRED
        0 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        1  [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        2  [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        3  [3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
        ...
        LABEL
        """
        distance_matrix = torch.abs(self.num_only_values.unsqueeze(0) - self.num_only_values.unsqueeze(1))
        return distance_matrix

    def _get_num_nan_vocab(self, tokenizer):
        """
        returns:
            tensor of shape (vocab_size,) where digits are mapped to their float value and other tokens are mapped to
            NaN.
        """

        # retrieve vocab from the tokenizer
        vocab = tokenizer.get_vocab()
        num_nan_vocab = torch.full((len(vocab),), float("nan"))

        # build vocab mapping from the vocab dictionary
        for string, token_id in vocab.items():
            stripped_token = string.strip()

            try:
                token_value = float(stripped_token)
                single_digit = -1 <= token_value <= 9 and len(stripped_token) == 1  # paper default: single digit only

                if self.multi_digits or single_digit:
                    num_nan_vocab[token_id] = token_value

            except ValueError:
                # print(f"!!! Parsing failed: Could not convert token {string} to float. Skipping...")
                continue

        return num_nan_vocab

    # pretty much following the code
    def _calc_ntl_was(self, logits, labels, importance_mask=None, ignore_index=-100):
        """
        logits (Tensor): logits of the model, shape (b, seq_len, vocab_size)
        labels (Tensor): labels of the model, shape (b, seq_len)
        importance_mask (Tensor): mask to weight number tokens by defined importance (not necessarily a binary mask),
        shape (b, seq_len)
        ignore_index (int): Token ID to ignore. default: -100

        returns:
            loss (Tensor): mean ntl-was loss for a batch, shape (1,)
        """
        # setting padding/ignored tokens' value to NaN, by switching their token ID to 0 (which has a value of NaN in
        # num_nan_vocab)
        labels = labels.masked_fill(labels == ignore_index, 0)
        # creating mask for number tokens in the labels
        all_labels_values = self.num_nan_vocab[labels]
        valid_labels_mask = ~torch.isnan(all_labels_values)
        valid_labels_values = all_labels_values[valid_labels_mask]

        # mask to weight the loss of number tokens based on their importance
        if importance_mask is not None:
            labels_imp_mask = importance_mask[valid_labels_mask]

        # if there's no valid labels (number tokens) or all tokens are masked/unimportant:
        #  we return the loss as 0 (ie no loss added to the main one)
        if not valid_labels_mask.any() or (importance_mask is not None and not labels_imp_mask.any()):
            return 0.0

        # --- Loss calculation ---
        # get probs of predicted number tokens
        number_logits = logits[:, :, self.num_tokens_mask]
        number_probs = F.softmax(number_logits, dim=-1)

        # Equation 4 in the paper:
        # they use absolute difference between the true/label numbers and all possible number values (vocab), weighted
        # by their probs. They leverage the fact that labels are one-hot encoded which is a simplified case of WAS. 
        # No need to compute the difference of CDFs.

        if not self.multi_digits: # using small cached distance matrix
            label_indices = valid_labels_values.long() # convert to int to use as indices
            distances_to_labels = self.distance_matrix[label_indices]
        else:
            distances_to_labels = torch.abs(
                valid_labels_values.unsqueeze(-1) - self.num_only_values
            )
        # shape: (num_valid_tokens,)
        valid_tokens_loss = (distances_to_labels * number_probs[valid_labels_mask]).sum(-1)

        if importance_mask is not None:
            # normalized by the number of tokens that have a non-zero weight
            batch_loss = (valid_tokens_loss * labels_imp_mask).sum() / labels_imp_mask.count_nonzero()
        else:
            batch_loss = valid_tokens_loss.mean()

        return batch_loss

    def __call__(self, *args, **kwargs):
        return self._calc_ntl_was(*args, **kwargs)
