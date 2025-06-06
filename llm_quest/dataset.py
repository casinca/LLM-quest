import gzip
import json

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms

from llm_quest.utils import alpaca_prompt_format


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

        for i in range(0, len(self.ids) - max_length, stride):
            self.input.append(self.ids[i : i + max_length])
            self.target.append(self.ids[i + 1 : i + max_length + 1])

        # convert to tensors
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
    InstructionDataset is a custom PyTorch Dataset for preparing instruction tuning data:
    formatting to Alpaca + tokenizing

    Args:
        file (str): Path to the JSON file  containing instruction examples with 'instruction',
                    'input', and 'output' keys.
        tokenizer (Tokenizer): The tokenizer object used to encode the text.

    Attributes:
        instruct_ids_list (list): List of tokenized instruction sequences.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized instruction sequence at the given index.
    """

    def __init__(self, file, tokenizer):
        with open(file, "r") as f:
            text = json.load(f)

        self.instruct_ids_list = []
        for instruct in text:
            formatted_instruct = alpaca_prompt_format(instruct)
            self.instruct_ids_list.append(tokenizer.encode(formatted_instruct))

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

    Attributes:
        instruct_ids_list (list): List of dictionaries, each containing tokenized prompt, chosen,
                                    and rejected responses.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the tokenized prompt, chosen, and rejected responses at the given index.

    """

    def __init__(self, file, tokenizer):
        with open(file, "r") as f:
            text = json.load(f)

        self.instruct_ids_list = []
        for instruct in text:
            # convert to alpaca format instructions + chosen & rejected responses (which include prompt too, hence full)
            formatted_instruct = alpaca_prompt_format(instruct, include_output=False)
            chosen_response = instruct["chosen"]
            rejected_response = instruct["rejected"]
            formatted_chosen_full = f"{formatted_instruct}\n\n### Response:\n{chosen_response}"
            formatted_rejected_full = f"{formatted_instruct}\n\n### Response:\n{rejected_response}"
            # tokenize
            tokenized_formatted_instruct = tokenizer.encode(formatted_instruct)
            tokenized_chosen_full = tokenizer.encode(formatted_chosen_full)
            tokenized_rejected_full = tokenizer.encode(formatted_rejected_full)

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

    return torch.tensor(padded_inputs).to(device), torch.tensor(padded_targets).to(device)


def custom_collate_fn(batch, pad_token_id=50256, allowed_max_length=None, mask_prompt_tokens=True, device="cpu"):
    """
    Copy of @rasbt's custom collate function for alignment finetuning
    """
    # Initialize lists to hold batch data
    batch_data = {"prompt": [], "chosen": [], "rejected": [], "rejected_mask": [], "chosen_mask": []}

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence) :] = False

            # Set mask for all input tokens to False
            # +2 sets the 2 newline ("\n") tokens before "### Response" to False
            if mask_prompt_tokens:
                mask[: prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data


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
