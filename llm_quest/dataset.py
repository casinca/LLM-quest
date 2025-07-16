import gzip
import json

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

from llm_quest.utils import ResponseExtractor, alpaca_deepseek_format, alpaca_prompt_format


class GPTDataset(Dataset):
    """
    GPTDataset is a custom PyTorch Dataset for preparing text data for training the GPT model.

    Args:
        text (str): The input text to be tokenized and transformed into sequences.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.
        max_length (int): The length of the input sequences.
        stride (int): The sentence overlap factor (sliding window)

    Attributes:
        input (list): A list of input sequences.
        target (list): A list of target sequences, which are the next tokens following the input sequences.
        ids (list): The tokenized representation of the input text.

    Methods:
        __len__(): Returns the number of input sequences.
        __getitem__(index): Returns the input and target sequences at the given index.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()

        self.input = []
        self.target = []

        self.ids = tokenizer.encode(text)

        # sliding window to create sequences
        for i in range(0, len(self.ids) - max_length, stride):
            self.input.append(self.ids[i : i + max_length])
            self.target.append(self.ids[i + 1 : i + max_length + 1])

        self.input = torch.tensor(self.input)
        self.target = torch.tensor(self.target)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]


class GPTStreamDataset(IterableDataset):
    """
    GPTStreamDataset is a class designed to handle streaming with HF datasets, ie large datasets that do not fit
    in memory.

    Args:
        stream (dict): An iterable data stream, where each element is expected to have a "text" field.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.
        max_length (int): The maximum length of the input sequences.
        stride (int): The overlap between consecutive sequences (sliding window).

    Yields:
        tuple: A tuple containing the input sequence and the target sequence as PyTorch tensors.
    """

    def __init__(self, stream, tokenizer, max_length, stride):
        super().__init__()
        self.stream = stream
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        for sample in self.stream:
            if "text" not in sample:
                raise KeyError('Missing "text" key in sample.')

            text = sample["text"]  # access the "text" field

            if not isinstance(text, str):
                print(f"Skipped: Unexpected type for text: {type(text)}")
                continue  # skip this sample if it's not a string

            ids = self.tokenizer.encode(text)

            for i in range(0, len(ids) - self.max_length, self.stride):
                input_seq = ids[i : i + self.max_length]
                target_seq = ids[i + 1 : i + self.max_length + 1]

                yield torch.tensor(input_seq), torch.tensor(target_seq)


class SpamDataset(Dataset):
    """
    SpamDataset is a custom PyTorch Dataset for preparing spam classification data.
    It also serves as a collate function as we are padding inputs & also create attention mask to:
    - retrieve the last token's logits
    - as padding mask

    Args:
        file (str): Path to the CSV file containing spam data with 'text' and 'label' columns.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.
        max_length (int, optional): Maximum length for padding/truncating sequences.
                                    If None, uses the longest sequence length.
        pad_token (int, optional): Token ID used for padding. Defaults to 50256 (GPT-2 padding token).

    Attributes:
        data (DataFrame): The loaded spam dataset.
        ids (list): List of tokenized text sequences.
        input (Tensor): Padded input sequences as a tensor.
        target (Tensor): Classification labels as a tensor.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the input sequence and label at the given index.
    """

    def __init__(self, file, tokenizer, max_length=None, pad_token=50256):
        super().__init__()

        self.data = pd.read_csv(file)
        self.ids = [tokenizer.encode(text) for text in self.data["text"]]
        # setting max_length as attribute to reuse for consistent max_length batching with val and test sets
        self.max_length = max_length

        # determining max_length for padding
        if self.max_length:
            self.ids = [id_vec[: self.max_length] for id_vec in self.ids]
        else:
            self.max_length = max(len(id_vec) for id_vec in self.ids)

        # padding manually (alt: torch.nn.utils.rnn.pad_sequence)
        padded_ids = [id_vec + [pad_token] * (self.max_length - len(id_vec)) for id_vec in self.ids]

        # attention_mask
        self.attention_mask = [[1] * len(id_vec) + [0] * (self.max_length - len(id_vec)) for id_vec in self.ids]

        # convert to tensors
        self.input = torch.tensor(padded_ids)
        self.target = torch.tensor(self.data["label"])
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.bool)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index], self.attention_mask[index]


class InstructionDataset(Dataset):
    """
    InstructionDataset is a custom PyTorch Dataset designed for loading, formatting, and tokenizing
    instruction-following data. It supports different file formats (JSON, JSONL) and allows for custom formatting
    functions.

    Args:
        file (str): Path to the dataset file. This can be a JSON file (e.g., Alpaca format) or a JSONL file (gsm8k),
                    containing full instruction examples.
        tokenizer (Tokenizer): The tokenizer object (e.g., from `tiktoken`) used to encode the text.
        formatting_func (callable, optional): The function used to format the instruction examples:
            - alpaca_prompt_format: for Alpaca format
            - alpaca_deepseek_format: for Alpaca + DeepSeek R1 reasoning format
            Defaults to `alpaca_prompt_format` (backward compatibility).
        file_type (str, optional): The format of the input file. Supported values are "json"
                                        and "jsonl". Defaults to "json".

    Attributes:
        instruct_ids_list (list): A list where each element is a list of token IDs, representing
                                    a tokenized and formatted instruction sequence.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized instruction sequence at the given index.

    Returns:
        list[int]: A list of token IDs representing a tokenized and formatted instruction sequence, ie:
        instruction + input (if any) + response
    """

    def __init__(self, file, tokenizer, formatting_func=alpaca_prompt_format, file_type="json"):
        self.instruct_ids_list = []

        if file_type == "json":
            with open(file, "r", encoding="utf-8") as f:
                text = json.load(f)
                for instruct in text:
                    formatted_instruct = formatting_func(instruct)
                    self.instruct_ids_list.append(tokenizer.encode(formatted_instruct))

        elif file_type == "jsonl":
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)
                    formatted_instruct = formatting_func(text)
                    self.instruct_ids_list.append(tokenizer.encode(formatted_instruct))
        else:
            raise ValueError(f"Invalid file type: {file_type}")

    def __len__(self):
        return len(self.instruct_ids_list)

    def __getitem__(self, index):
        return self.instruct_ids_list[index]


class HFDataset(Dataset):
    """
    HFDataset is a custom PyTorch Dataset designed for loading and tokenizing text data
    from a compressed JSONL file (typically sourced from Hugging Face datasets.)

    Args:
        file (str): Path to the compressed JSONL file (.jsonl.gz) containing the text data.
                    Each line should be a valid JSON object with a "text" key.
        tokenizer (Tokenizer): The tokenizer object used to encode the text data.
        max_samples (int, optional): The maximum number of samples to load from the dataset.
                                    If None, loads all samples. Defaults to None.

    Attributes:
        texts (list): List of tokenized text sequences.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized text sequence at the given index.
    """

    def __init__(self, file, tokenizer, max_samples=None):
        self.texts = []
        sample_count = 0

        # read the compressed JSONL file
        with gzip.open(file, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # tokenize the text and add it to the list
                self.texts.append(tokenizer.encode(data["text"]))
                # stop if max_samples is reached
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


# TODO for now we hardcode, HF dataset has different names for the labels keys + values for standardization
class ImageDataset(Dataset):
    """
    Custom dataset class that converts PIL images from HF datasets to tensors and applies transforms.

    Args:
        hf_dataset_split (dict): Hugging Face dataset split, from the load_dataset(),
                                ex: dataset["train"], dataset["test"]...
        standardize(bool): Whether to standardize the images
    """

    def __init__(self, hf_dataset_split, standardize=False):
        self.dataset = hf_dataset_split

        if standardize:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # this is hardcoded for CIFAR-10:
                    #  https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                ]
            )
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]  # PIL Image
        label = item["label"]  # hardcoded

        image = self.transform(image)  # Convert and normalize the image to a tensor

        return image, label


class PreferenceDataset(Dataset):
    """
    PreferenceDataset is a custom PyTorch Dataset for preparing preference tuning data, similar to InstructionDataset.

    Args:
        file (str): Path to the JSON file containing preference examples with 'instruction',
                    'input', 'chosen', and 'rejected' keys.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.
        prompts_only (bool, optional): Whether to return only the prompts or not. (useful for RLHF training)

    Attributes:
        instruct_ids_list (list): List of dictionaries, each containing tokenized prompt, chosen,
                                    and rejected responses.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized prompt, chosen, and rejected responses at the given index.

    Returns:
        dict[str, list[int]]: A dictionary containing:
            - prompt: Tokenized prompt (instruction+question).
            - chosen: Tokenized prompt + chosen response.
            - rejected: Tokenized prompt + rejected response.

    """

    def __init__(self, file, tokenizer, prompts_only=False):
        with open(file, "r", encoding="utf-8") as f:
            text = json.load(f)

        self.instruct_ids_list = []
        for instruct in text:
            # convert to alpaca format instructions + chosen & rejected responses (which include prompt too, hence full)
            formatted_instruct = alpaca_prompt_format(instruct, include_output=False)
            chosen_response = instruct["chosen"]
            rejected_response = instruct["rejected"]
            formatted_chosen_full = f"{formatted_instruct}{chosen_response}"
            formatted_rejected_full = f"{formatted_instruct}{rejected_response}"
            # tokenize
            tokenized_formatted_instruct = tokenizer.encode(formatted_instruct)
            tokenized_chosen_full = tokenizer.encode(formatted_chosen_full)
            tokenized_rejected_full = tokenizer.encode(formatted_rejected_full)

            if prompts_only:
                self.instruct_ids_list.append(tokenized_formatted_instruct)
            else:
                self.instruct_ids_list.append(
                    {
                        "prompt": tokenized_formatted_instruct,
                        "chosen": tokenized_chosen_full,
                        "rejected": tokenized_rejected_full,
                    }
                )

    def __len__(self):
        return len(self.instruct_ids_list)

    def __getitem__(self, index):
        return self.instruct_ids_list[index]


class ReasoningDataset(Dataset):
    """
    ReasoningDataset is a custom PyTorch Dataset for preparing reasoning data, similar to InstructionDataset.

    Args:
        file (str): Path to the JSONL file containing reasoning examples with 'question' and 'answer' keys.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.

    Attributes:
        instruct_ids_list (list): List of dictionaries, each containing tokenized prompt, full response,
                                    and the extracted answer.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized prompt, full response, and extracted answer at the given index.

    Returns:
        dict[str, list[int]]: A dictionary containing:
            - prompt: Tokenized prompt (instruction+question).
            - full_response: Tokenized prompt + full generated response (including reasoning and answer tags).
            - answer: Tokenized final answer extracted from the full response.

    """

    def __init__(self, file, tokenizer):
        self.instruct_ids_list = []
        with open(file, "r") as f:
            for line in f:
                text = json.loads(line)

            # convert to alpaca format: instructions + responses with reasoning & answer tags format
            formatted_reasoning = alpaca_deepseek_format(text, include_response=True)
            prompt, full_response, answer = self._get_prompt_response_answer(formatted_reasoning)

            # tokenize
            tokenized_prompt = tokenizer.encode(prompt)
            tokenized_full_response = tokenizer.encode(full_response)
            tokenized_answer = tokenizer.encode(answer)

            self.instruct_ids_list.append(
                {
                    "prompt": tokenized_prompt,
                    "full_response": tokenized_full_response,
                    "answer": tokenized_answer,
                }
            )

    def _get_prompt_response_answer(self, formatted_text):
        """
        helper function to get the prompt, full_response: prompt + response (including answer), and answer from the
        formatted reasoning text.
        """
        prompt, sep, response = formatted_text.partition("### Response:")
        prompt = prompt + sep  # prompt also include "### Response:"
        full_response = response.strip()
        answer = ResponseExtractor.get_answer(full_response)

        return prompt, full_response, answer

    def __len__(self):
        return len(self.instruct_ids_list)

    def __getitem__(self, index):
        return self.instruct_ids_list[index]


# We don't necessarily need to have attention masks because the no_loss tokens will mask padded tokens
# during loss calc anyways.
# Yes it's a bit of overhead but for good practice: preventing the model from paying attention from/to padded tokens.
# We could have used attn_mask as a mask for the loss too and CE arg reduce="None", but PyTorch CE function with
# built-in detection of -100 as no_loss token is standard practice and more efficient.
#
# O(3N) list comprehension was faster when benchmarking than dispatching in a single for loop in O(N)
def collate_function(batch, custom_max_len=None, device="cpu"):
    """
    Custom collate function for batching sequences of variable lengths used with InstructionDataset class.
    Following "instruction Tuning With Loss Over Instructions" paper, ie not masking instructions during loss

    Args:
        batch (list): List of lists to be batched together
        custom_max_len (int, optional): Maximum allowed sequence length. If provided,
            sequences will be truncated to this length. Defaults to None.
        device (str, optional): Device to place the tensors on. Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Padded input sequences
            - torch.Tensor: Padded target sequences shifted by 1 position
            - torch.Tensor: Attention mask (1 for real tokens, 0 for padding)
    """
    pad_token_id = 50256  # id of the "<|endoftext|>" token
    no_loss_id = -100  # default Pytorch Cross Entropy ignore_index=-100 for ignoring this token id during loss calc

    # truncate each sequence if custom_max_len is provided
    if custom_max_len:
        truncated_batch = [sample[:custom_max_len] for sample in batch]
    else:
        truncated_batch = batch

    batch_max_len = max(len(sample) for sample in truncated_batch)

    # padding a batch's inputs to the max sample's length it contains
    # (each batch will have their own custom len rather than a fixed universal dataset len)
    padded_inputs = [sample + [pad_token_id] * (batch_max_len - len(sample)) for sample in truncated_batch]
    # similarly for targets: shifting by 1, but adding only 1 padding token, rest is filled of "don't compute loss"
    # tokens
    padded_targets = [
        sample[1:] + [pad_token_id] + [no_loss_id] * (batch_max_len - len(sample)) for sample in truncated_batch
    ]
    # attention masks: 1 for real tokens, 0 for padding
    attention_masks = [[1] * len(sample) + [0] * (batch_max_len - len(sample)) for sample in truncated_batch]

    return (
        torch.tensor(padded_inputs).to(device),
        torch.tensor(padded_targets).to(device),
        torch.tensor(attention_masks, dtype=torch.bool).to(device),
    )


# Alternative experimental collate scheme where:
#
# Special tokens(EoS here): "attend & don't compute loss" vs "don't attend & don't compute loss" as in
# collate_function()
#
# This doesn't change anything for the loss, it's just for the attention masks.
#
# based on:
# https://discuss.huggingface.co/t/difference-between-setting-label-index-to-100-setting-attention-mask-to-0/4503/4
# https://github.com/huggingface/trl/issues/1623
def collate_function_eos(batch, custom_max_len=None, device="cpu"):
    """
    Same as `collate_function()` but with a different attention mask scheme, where EoS token is not masked.

    Args:
        batch (list): List of lists to be batched together
        custom_max_len (int, optional): Maximum allowed sequence length. If provided,
            sequences will be truncated to this length. Defaults to None.
        device (str, optional): Device to place the tensors on. Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Padded input sequences
            - torch.Tensor: Padded target sequences shifted by 1 position
            - torch.Tensor: Attention mask (1 for real tokens, 0 for padding)
    """
    pad_token_id = 50256  # id of the "<|endoftext|>" token
    no_loss_id = -100  # default Pytorch Cross Entropy ignore_index=-100 for ignoring this token id during loss calc

    # truncate each sequence if custom_max_len is provided
    if custom_max_len:
        truncated_batch = [sample[: custom_max_len - 1] for sample in batch]
    else:
        truncated_batch = batch

    # add 1 EoS token to the end of each sequence
    truncated_batch = [sample + [pad_token_id] for sample in truncated_batch]

    batch_max_len = max(len(sample) for sample in truncated_batch)

    # padding a batch's inputs to the max sample's length it contains
    # (each batch will have their own custom len rather than a fixed universal dataset len)
    padded_inputs = [sample + [pad_token_id] * (batch_max_len - len(sample)) for sample in truncated_batch]
    # similarly for targets: shifting by 1, but adding only 1 padding token, rest is filled of "don't compute loss"
    # tokens
    padded_targets = [sample[1:] + [no_loss_id] * (batch_max_len - len(sample) + 1) for sample in truncated_batch]
    # attention masks: 1 for real tokens, 0 for padding
    attention_masks = [[1] * len(sample) + [0] * (batch_max_len - len(sample)) for sample in truncated_batch]

    return (
        torch.tensor(padded_inputs).to(device),
        torch.tensor(padded_targets).to(device),
        torch.tensor(attention_masks, dtype=torch.bool).to(device),
    )


def dpo_collate(batch, pad_token_id=50256, allowed_max_length=None, mask_prompt_tokens=True, device="cpu"):
    """
    Custom collate function for Direct Preference Optimization (DPO) training.

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries, where each dictionary represents a single data
        sample and contains the following keys:
                - "prompt" (list[int]): List of token IDs for the prompt.
                - "chosen" (list[int]): List of token IDs for the chosen response (including prompt).
                - "rejected" (list[int]): List of token IDs for the rejected response (including prompt).
        pad_token_id (int, optional): The token ID used for padding sequences to a common length.
                                    Defaults to 50256 (GPT-2 EOS token).
        allowed_max_length (int, optional): If specified, sequences will be truncated to this maximum length
                                            before padding.
        mask_prompt_tokens (bool, optional): Whether to mask the prompt tokens.
        device (str or torch.device, optional): The device ("cpu" or "cuda") to which the output tensors
                                                should be moved. Defaults to "cpu".

    Returns:
        dict: A dictionary containing four PyTorch tensors:
            - "chosen" (torch.Tensor): Padded chosen sequences. Shape: (batch_size, max_length_common).
            - "rejected" (torch.Tensor): Padded rejected sequences. same shape as chosen.
            - "chosen_mask" (torch.Tensor): Boolean loss mask for chosen sequences. True for real sequence tokens,
                                            False for padding tokens and optionally prompt tokens. same shape as chosen.
            - "rejected_mask" (torch.Tensor): Boolean loss mask for rejected sequences. same shape as chosen_mask.

    """

    # Determine the longest sequence to set a common padding length
    if batch:
        max_chos_len = max(len(item["chosen"]) for item in batch)
        max_rej_len = max(len(item["rejected"]) for item in batch)
        max_length_common = max(max_chos_len, max_rej_len) + 1  # +1 for shifting labels

    if allowed_max_length is not None:
        max_length_common = min(max_length_common, allowed_max_length)

    bsz = len(batch)
    # preallocating batch tensors
    batch_chosen = torch.full((bsz, max_length_common), fill_value=pad_token_id, dtype=torch.long, device=device)
    batch_chosen_mask = torch.ones(bsz, max_length_common, dtype=torch.bool, device=device)
    batch_rejected = batch_chosen.clone()
    batch_rejected_mask = batch_chosen_mask.clone()

    # Process each item in the batch
    for i, item in enumerate(batch):
        prompt_len = len(item["prompt"])

        chos = item["chosen"]
        rej = item["rejected"]
        # truncate if needed (before padding, more efficient)
        if allowed_max_length is not None:
            chos = chos[:max_length_common]
            rej = rej[:max_length_common]

        chos_len = len(chos)
        rej_len = len(rej)

        batch_chosen[i, :chos_len] = torch.tensor(chos, dtype=torch.long)
        batch_rejected[i, :rej_len] = torch.tensor(rej, dtype=torch.long)

        batch_chosen_mask[i, chos_len:] = False
        batch_rejected_mask[i, rej_len:] = False

        if mask_prompt_tokens:
            batch_chosen_mask[i, :prompt_len] = False
            batch_rejected_mask[i, :prompt_len] = False

    return {
        "chosen": batch_chosen.to(device),
        "rejected": batch_rejected.to(device),
        "chosen_mask": batch_chosen_mask.to(device),
        "rejected_mask": batch_rejected_mask.to(device),
    }


# similar to dpo_collate, with additional attn masks and more efficient
def reward_pref_collate(batch, pad_token_id=50256, allowed_max_length=None, device="cpu"):
    """
    Custom collate function for the Reward Model training with preference data.
    It prepares chosen and rejected sequences, along with their loss and attention masks.

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries, where each dictionary represents a single data
                                            sample and contains the following keys:
                                - "prompt" (list[int]): List of token IDs for the prompt.
                                - "chosen" (list[int]): List of token IDs for the chosen response (including prompt).
                                - "rejected" (list[int]): List of token IDs for the rejected response (including prompt).
        pad_token_id (int, optional): The token ID used for padding sequences to a common length.
                                    Defaults to 50256 (GPT-2 EOS token).
        allowed_max_length (int, optional): If specified, sequences will be truncated to this maximum length
                                            before padding.
        device (str or torch.device, optional): The device ("cpu" or "cuda") to which the output tensors
                                                should be moved. Defaults to "cpu".

    Returns:
        dict: A dictionary containing six PyTorch tensors:
            - "chosen" (torch.Tensor): Padded chosen sequences. Shape: (batch_size, max_length_common).
            - "rejected" (torch.Tensor): Padded rejected sequences. Same shape as chosen.
            - "chosen_mask" (torch.Tensor): Boolean loss mask for chosen sequences. True for real response tokens+EoS,
                                            False for padding tokens and prompt tokens. same shape as chosen.
            - "rejected_mask" (torch.Tensor): Boolean loss mask for rejected sequences. same shape as chosen.
            - "chosen_attn_mask" (torch.Tensor): Boolean attention mask for chosen sequences. True for real tokens,
                                                False for padding tokens. Same shape as chosen.
            - "rejected_attn_mask" (torch.Tensor): Boolean attention mask for rejected sequences. Same shape as chosen
    """
    # Determine the longest sequence to set a common padding length
    max_chos_len = max(len(item["chosen"]) for item in batch)
    max_rej_len = max(len(item["rejected"]) for item in batch)
    max_length_common = max(max_chos_len, max_rej_len) + 1  # here the +1 is for the EoS token, not for shifting labels

    if allowed_max_length is not None:
        max_length_common = min(max_length_common, allowed_max_length)

    bsz = len(batch)
    # Preallocate tensors
    batch_chosen = torch.full((bsz, max_length_common), fill_value=pad_token_id, dtype=torch.long, device=device)
    batch_rejected = batch_chosen.clone()

    # Lists to store lengths for vectorized mask creation
    prompt_lens, chos_lens, rej_lens = [], [], []

    # Process each item in the batch
    for i, item in enumerate(batch):
        prompt_len = len(item["prompt"])
        chos = item["chosen"] + [pad_token_id]  # adding EoS token to all sequences
        rej = item["rejected"] + [pad_token_id]

        if allowed_max_length is not None:
            chos = chos[:max_length_common]
            rej = rej[:max_length_common]

        chos_len = len(chos)
        rej_len = len(rej)

        batch_chosen[i, :chos_len] = torch.tensor(chos, dtype=torch.long)
        batch_rejected[i, :rej_len] = torch.tensor(rej, dtype=torch.long)

        prompt_lens.append(prompt_len)
        chos_lens.append(chos_len)
        rej_lens.append(rej_len)

    # vectorized mask creation
    prompt_lens_t = torch.tensor(prompt_lens, device=device)
    chos_lens_t = torch.tensor(chos_lens, device=device)
    rej_lens_t = torch.tensor(rej_lens, device=device)

    # tensor of indices [0, 1, ..., max_len-1] to use as mask
    indices = torch.arange(max_length_common, device=device).expand(bsz, -1)

    # Attention masks are True where indices are less than the sequence length
    batch_chosen_attn_mask = indices < chos_lens_t.unsqueeze(1)
    batch_rejected_attn_mask = indices < rej_lens_t.unsqueeze(1)

    # reward masks are True where indices are >= prompt_len AND also part of the real sequence EoS included
    prompt_mask = indices >= prompt_lens_t.unsqueeze(1)
    batch_chosen_mask = prompt_mask & batch_chosen_attn_mask
    batch_rejected_mask = prompt_mask & batch_rejected_attn_mask

    return {
        "chosen": batch_chosen.to(device),
        "rejected": batch_rejected.to(device),
        "chosen_mask": batch_chosen_mask.to(device),
        "rejected_mask": batch_rejected_mask.to(device),
        "chosen_attn_mask": batch_chosen_attn_mask.to(device),
        "rejected_attn_mask": batch_rejected_attn_mask.to(device),
    }


def create_dataloader(
    text,
    batch_size,
    max_length,
    stride,
    tokenizer,
    shuffle,
    drop_last,
    num_workers,
    streaming=False,
    pin_memory=False,
):
    """
    Wrapper
    Creates a DataLoader for the GPTDataset or GPTStreamDataset.

    Args:
        text (str/dict): The input text to be processed: either as a string or a stream.
        batch_size (int): The size of each batch in the DataLoader.
        max_length (int): The maximum length of each sequence in the dataset.
        stride (int): The stride for creating overlapping sequences.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.
        shuffle (bool): Whether to shuffle the dataset.
        drop_last (bool): Whether to drop the last batch if it's not a full batch.
        num_workers (int): The number of workers to use for data loading.
        pin_memory (bool): Whether to pin memory for faster transfer to GPU.
        streaming (bool): Whether to use streaming for large datasets.

    Returns:
        DataLoader: A DataLoader instance for the GPTDataset.
    """
    if streaming:
        dataset = GPTStreamDataset(text, tokenizer, max_length, stride)
    else:
        dataset = GPTDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


# testing code
# if __name__ == "__main__":
#    # LOADER PART
#
#    with open("data/the-verdict.txt", "r") as file:
#        txt = file.read()
#
#    batch_size = 8
#    n_ctx = max_length = 4
#    vocab_size = 50257
#    emb_len = 256
#
#    tokenizer = tiktoken.get_encoding("gpt2")
#
#    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=4)
#
#    print(dataset[0:2], "\n")
#
#    dataloader = create_dataloader(
#        txt,
#        batch_size=batch_size,
#        max_length=max_length,
#        stride=2,
#        tokenizer=tokenizer,
#        shuffle=False,
#        drop_last=True,
#        num_workers=0,
#    )
#
#    # Print only the first sample
#    input_tensor, target_tensor = next(iter(dataloader))
#    print(input_tensor, target_tensor)
#
#    embed_dict = torch.nn.Embedding(vocab_size, emb_len)
#    input, target = next(iter(dataloader))
#    print(input, input.shape)
#
#    raw_embedding = embed_dict(input)
#    print(raw_embedding, raw_embedding.shape)
#
#    pos_embed_dict = torch.nn.Embedding(n_ctx, emb_len)
#    pos_embedding = pos_embed_dict(torch.arange(n_ctx))
#    print(pos_embedding.shape)
#
#    embedding = raw_embedding + pos_embedding
#    print(embedding.shape)
#
#   tokenizer = tiktoken.get_encoding("gpt2")
#   train_dataset = SpamDataset(file="../data/spam_preprocessed/train.csv", max_length=300, tokenizer=tokenizer)
#
#   print(len(train_dataset.input[0]))
