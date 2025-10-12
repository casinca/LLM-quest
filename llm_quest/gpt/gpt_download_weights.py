import os
from pathlib import Path

import requests
import torch
from tqdm import tqdm


def download_gpt_model(gpt_size, save_dir):
    """
    Downloads, verifies, and saves GPT-2 weights to the specified directory.

    This script will not create any extra subdirectories. The file will be
    saved directly inside the path provided.

    Args:
        gpt_size (str): The size of the gpt model. It can be one of the following:
            "gpt_s"  → gpt2-small (124M)
            "gpt_m"  → gpt2-medium (355M)
            "gpt_l"  → gpt2-large (774M)
            "gpt_xl" → gpt2-xl (1558M)
        save_directory (str or Path): The exact folder where the model file will be saved.

    Returns:
        Path to the downloaded model file (if successful, None otherwise)
    """
    model_filenames = {
        "gpt_s": "gpt2-small-124M.pth",
        "gpt_m": "gpt2-medium-355M.pth",
        "gpt_l": "gpt2-large-774M.pth",
        "gpt_xl": "gpt2-xl-1558M.pth",
    }

    if gpt_size not in model_filenames:
        print(f"Error: Invalid gpt_size '{gpt_size}'.")
        print("Choose from: 'gpt_s', 'gpt_m', 'gpt_l', 'gpt_xl'")
        return

    filename = model_filenames[gpt_size]

    # Path setup
    # use the provided directory as the final destination without modification.
    save_dir = Path(save_dir)
    save_path = save_dir / filename
    # Credit to @rasbt for hosting the files on Hugging Face
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{filename}"

    # ensure the user-specified directory exists.
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Use a streaming GET request to get headers of the final file
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            remote_size = int(response.headers.get("content-length", 0))

            # Check if file already exists, is valid, and skip download if so
            if save_path.exists():
                local_size = save_path.stat().st_size
                if local_size == remote_size and remote_size > 0:
                    print(f"'{filename}' already exists in '{save_dir}' and is valid.")
                    print("Skipping download.")
                    return save_path
                else:
                    print(f"File exists but is invalid. Local: {local_size}, Remote: {remote_size}. Re-downloading.")

            # Download the file if not present or invalid
            print(f"Downloading {filename} to {save_path}...")
            with open(save_path, "wb") as f, tqdm(
                total=remote_size,
                unit="B",
                unit_scale=True,
                desc=filename,
            ) as pbar:

                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"Successfully downloaded to {save_path}")
            return save_path

    # If nothing worked, remove incomplete directory
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if save_path.exists():
            # check size before deleting to be safe
            if save_path.stat().st_size != remote_size:
                os.remove(save_path)
                print(f"Removed incomplete file: {save_path}")

        return None


def load_gpt_weights(model, weights_path):
    """
    Loads weights from a .pth file, remaps key names to match my custom
    GPT model weights name, and loads them into the model.

    Args:
        model (torch.nn.Module): The GPTModel instance.
        weights_path (str): The file path to the downloaded .pth model weights.
    """

    pretrained_weights = torch.load(weights_path)
    model_state_dict = model.state_dict()
    remapped_weights = {}

    # Remapping tuples as (pattern_to_find, replacement)
    # order matters: exact matches first, then substring replacements for layers idx
    remapping_rules = [
        # Exact matches for blocks with a single weight: embedding layers + output head
        ("tok_emb.weight", "emb_dict.weight"),
        ("pos_emb.weight", "pos_emb_dict.weight"),
        ("out_head.weight", "out.weight"),
        # Partial match for blocks with multiple weight: trf layers, norms (these will be applied with .replace())
        ("att.W_query", "att.w_queries"),
        ("att.W_key", "att.w_keys"),
        ("att.W_value", "att.w_values"),
        (".norm1.", ".ln_1."),  # LayerNorm weights are scale + shift
        (".norm2.", ".ln_2."),
        ("final_norm.", "final_ln."),
        (".ff.", ".ffn."),
    ]

    print("\nStarting weight remapping...")
    for old_key, value in pretrained_weights.items():
        new_key = old_key

        # Remapping logic
        for pattern, replacement in remapping_rules:
            if pattern in new_key:
                new_key = new_key.replace(pattern, replacement)
                # break if it's an exact match (no need to iterate further over partial matches)
                if pattern == old_key:
                    break

        # Assignment
        if new_key in model_state_dict:
            # sanity check: ensure shapes match
            if model_state_dict[new_key].shape == value.shape:
                remapped_weights[new_key] = value
                # print(f"Mapped: {old_key}  ->  {new_key}")
            else:
                print(
                    f"WARNING Shape mismatch for key '{new_key}': "
                    f"Model expects {model_state_dict[new_key].shape}, "
                    f"Pre-trained has {value.shape}. Skipping."
                )
        else:
            print(f"WARNING Key '{new_key}' not found in model state_dict. Skipping.")

    # Load remapped weights into the model
    model.load_state_dict(remapped_weights)
    print("Weights successfully remapped and loaded into the model.\n")
