import torch


# adapted from my custom_collate in llm_quest/engine.py
def collate_function_mtp(batch, custom_max_len=None, k=2, device="cpu"):
    """
    Custom collate function for batching sequences of variable lengths together, suited for Multi-Token Prediction
    (MTP).

    This function not only prepares inputs & targets but also generates k-th shifted inputs & targets for k MTP
    modules.

    Args:
        batch (list): List of lists, where each inner list represents a sequence of token IDs.
        custom_max_len (int, optional): Maximum allowed sequence length. If provided, sequences will be truncated.
        k (int, optional): Number of MTP modules (and thus, the number of shifted input/target pairs to generate).
        device (str, optional): Device to place the tensors on ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Padded input sequences.
            - torch.Tensor: Padded target sequences (shifted by 1 position).
            - list[torch.Tensor]: List of k padded, shifted input sequences for MTP.
            - list[torch.Tensor]: List of k padded, shifted target sequences for MTP.
    """
    pad_token_id = 50256  # id of the "<|endoftext|>" token
    no_loss_id = -100  # default Pytorch Cross Entropy ignore_index=-100, this token will be ignored during loss calc

    # truncate each sequence if custom_max_len is provided
    if custom_max_len:
        truncated_batch = [sample[:custom_max_len] for sample in batch]
    else:
        truncated_batch = batch

    batch_max_len = max(len(sample) for sample in truncated_batch)

    # we are dropping sequence of length <= k to avoid edge cases
    padded_inputs = [
        sample + [pad_token_id] * (batch_max_len - len(sample)) for sample in truncated_batch if len(sample) > k
    ]

    padded_targets = [
        sample[1:] + [pad_token_id] + [no_loss_id] * (batch_max_len - len(sample))
        for sample in truncated_batch
        if len(sample) > k
    ]

    mtp_inputs, mtp_targets = [], []

    for i in range(1, k + 1):
        shifted_inputs = [
            sample[i:] + [pad_token_id] * (batch_max_len + i - len(sample))
            for sample in truncated_batch
            if len(sample) > k
        ]

        shifted_targets = [
            sample[i + 1 :] + [pad_token_id] + [no_loss_id] * (batch_max_len + i - len(sample))
            for sample in truncated_batch
            if len(sample) > k
        ]

        mtp_inputs.append(torch.tensor(shifted_inputs).to(device))
        mtp_targets.append(torch.tensor(shifted_targets).to(device))

    return (
        torch.tensor(padded_inputs).to(device),
        torch.tensor(padded_targets).to(device),
        mtp_inputs,
        mtp_targets,
    )


def collate_function_mtp_experimental(batch, custom_max_len=None, k=2, device="cpu"):
    """
    This collate function will instead shrink the inputs and targets by k

    ex: if k=2, seq=[1,2,3,4,5] → inputs=[1,2,3,P,P], targets=[2,3,P,N,N]
                                → shifted inputs1=[2,3,4,P,P], shifted targets1=[3,4,P,N,N]
                                → shifted inputs2=[3,4,5,P,P], shifted targets2=[4,5,P,N,N]

    """
    pad_token_id = 50256  # id of the "<|endoftext|>" token
    no_loss_id = -100  # default Pytorch Cross Entropy ignore_index=-100, this token will be ignored during loss calc

    # truncate each sequence if custom_max_len is provided
    if custom_max_len:
        truncated_batch = [sample[:custom_max_len] for sample in batch]
    else:
        truncated_batch = batch

    batch_max_len = max(len(sample) for sample in truncated_batch)

    padded_inputs = [
        sample[:-k] + [pad_token_id] * (batch_max_len + k - len(sample))
        for sample in truncated_batch
        if len(sample) > k
    ]

    padded_targets = [
        sample[1:-k] + [pad_token_id] + [no_loss_id] * (batch_max_len + k - len(sample))
        for sample in truncated_batch
        if len(sample) > k
    ]

    mtp_inputs, mtp_targets = [], []

    for i in range(1, k + 1):
        shifted_inputs = [
            sample[i : len(sample) - k + i] + [pad_token_id] * (batch_max_len + k - len(sample))
            for sample in truncated_batch
            if len(sample) > k
        ]

        shifted_targets = [
            sample[i + 1 : len(sample) - k + i] + [pad_token_id] + [no_loss_id] * (batch_max_len + k - len(sample))
            for sample in truncated_batch
            if len(sample) > k
        ]

        mtp_inputs.append(torch.tensor(shifted_inputs).to(device))
        mtp_targets.append(torch.tensor(shifted_targets).to(device))

    return (
        torch.tensor(padded_inputs).to(device),
        torch.tensor(padded_targets).to(device),
        mtp_inputs,
        mtp_targets,
    )


# test
if __name__ == "__main__":
    batch_example = [
        [1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 10, 11, 12],
    ]

    D = 2

    collated_batch = collate_function_mtp(batch_example, k=D)

    padded_inputs, padded_targets, mtp_inputs, mtp_targets = collated_batch

    print("Padded Inputs:")
    print(padded_inputs)

    print("\nPadded Targets:")
    print(padded_targets)

    print("\nMTP Inputs:")
    for i, mtp_input in enumerate(mtp_inputs):
        print(f"  Shift {i+1}:")
        print(f"    {mtp_input}")

    print("\nMTP Targets:")
    for i, mtp_target in enumerate(mtp_targets):
        print(f"  Shift {i+1}:")
        print(f"    {mtp_target}")

    print("\n")
    print("-------------------------------------")

    collated_batch2 = collate_function_mtp_experimental(batch_example, k=D)

    print("Original Batch:")
    for seq in batch_example:
        print(seq)

    print("\nCollated Batch:")
    print("  Main Inputs:\n", collated_batch2[0])
    print("  Main Targets:\n", collated_batch2[1])
    for i in range(D):
        print(f"  MTP Inputs (k={i+1}):\n", collated_batch2[2][i])
        print(f"  MTP Targets (k={i+1}):\n", collated_batch2[3][i])
