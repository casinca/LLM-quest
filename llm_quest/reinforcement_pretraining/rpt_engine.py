import torch

from llm_quest.utils import ResponseExtractor


class PrefixMatchingReward:
    """
    Reward calculator following the prefix matching strategy from the RPT paper:
    see 3.2 Pre-Training with Reinforcement Learning and Appendix A. Design Choices of Reward

    2 conditions must be met to get a positive reward:
        - the answer must be a prefix of the labels
        - the answer must be a valid boundary of the labels

        The valid boundary is the set of all possible byte lengths of the labels.

    args:
        tokenizer: the tokenizer to use to tokenize the labels
        good_answer_reward: the reward for a good answer
        wrong_answer_reward: the penalty for a wrong answer
        unfinished_answer_reward: the penalty for an unfinished answer

    """

    def __init__(
        self,
        tokenizer,
        good_answer_reward=1.0,
        wrong_answer_reward=0.0,
        unfinished_answer_reward=-10.0,
    ):
        assert wrong_answer_reward <= 0, "wrong_answer_reward should be ≤ 0"
        assert unfinished_answer_reward <= 0, "unfinished_answer_reward should be ≤ 0"

        self.tokenizer = tokenizer
        self.good_answer_reward = good_answer_reward
        self.wrong_answer_reward = wrong_answer_reward
        self.unfinished_answer_reward = unfinished_answer_reward

    @staticmethod
    def _is_prefix(answer_bytes, label_bytes):
        """
        Args:
            answer_bytes (bytes): The answer in bytes.
            label_bytes (bytes): The label in bytes.
        """
        return label_bytes.startswith(answer_bytes)

    @staticmethod
    def _is_valid_boundary(answer_bytes, valid_boundary):
        """
        Args:
            answer_bytes (bytes): The answer in bytes.
            valid_boundary (set[int]): The valid boundary of the labels.
        """
        return len(answer_bytes) in valid_boundary

    def _get_valid_boundary(self, label):
        """
        calc & return the valid boundary of the current label string.

        Args:
            label (str): The ground truth label.

        Returns:
            set[int]: The valid boundary of the label.
        """
        valid_boundary = set()
        token_ids = self.tokenizer.encode(label)

        for i in range(1, len(token_ids) + 1):
            token_id = token_ids[:i]
            token_string = self.tokenizer.decode(token_id)
            token_bytes = token_string.encode("utf-8")
            valid_boundary.add(len(token_bytes))

        return valid_boundary

    def _calc_reward(self, model_responses, labels):
        """
        Calculate the rewards based on the model's answers, for a batch.

        Args:
            model_responses (list[str]): The decoded model's responses.
            labels (list[str]): The ground truth labels.

        Returns:
            list[float]: The rewards for a batch.

        """
        rewards_list = []

        for response_string, label in zip(model_responses, labels):
            # we do not want to sanitize the answer here, unlike RLVR, ex: spaces are important for MTP
            model_answer = ResponseExtractor.get_answer(response_string)

            if model_answer is None:
                rewards_list.append(self.unfinished_answer_reward)
                continue

            valid_boundary = self._get_valid_boundary(label)
            # convert to bytes before checking both conditions
            answer_bytes = model_answer.encode("utf-8")
            label_bytes = label.encode("utf-8")

            # check both conditions
            if (
                PrefixMatchingReward._is_prefix(answer_bytes, label_bytes)
                and PrefixMatchingReward._is_valid_boundary(answer_bytes, valid_boundary)
            ):  # fmt: skip
                rewards_list.append(self.good_answer_reward)
            else:
                rewards_list.append(self.wrong_answer_reward)

        return rewards_list

    def __call__(self, model_responses, labels):
        """
        Main orchestrator for the rewards' calculation.

        Args:
            model_responses (torch.Tensor): The model's responses, shape (B, S)
            labels (list[str]): The ground truth labels.

        Returns:
            torch.Tensor: The total rewards for a batch of responses, shape (B,)
        """
        decoded_responses = self.tokenizer.batch_decode(model_responses, skip_special_tokens=True)
        rewards_list = self._calc_reward(decoded_responses, labels)

        return torch.tensor(rewards_list, dtype=torch.bfloat16, device=model_responses.device)
