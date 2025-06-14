import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from config import TINY_VIT_CONFIG
from llm_quest.dataset import ImageDataset
from llm_quest.vit.vit_engine import vit_training_eval_loop
from llm_quest.vit.vit_model import ViTModel

# --- Hyperparameters ---
batch_size = 256
num_epochs = 20
peak_lr = 8e-4
init_lr = 1e-6
min_lr = 1e-6
weight_decay = 0.3
warmup_percent = 0.3
eval_freq = 200
eval_iter = 25
num_workers = 10
pin_memory = True
persistent_workers = False
amp = True

if __name__ == "__main__":
    torch.manual_seed(123)

    # --- Loaders ---
    print("Loading dataset...")
    dataset = load_dataset("uoft-cs/cifar10")

    train_dataset = ImageDataset(dataset["train"], standardize=True)
    val_dataset = ImageDataset(dataset["test"], standardize=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- Model & Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")
    model = ViTModel(TINY_VIT_CONFIG)
    model.to(device)

    # model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViT model created with {total_params:,} parameters")
    print(f"Model configuration: {TINY_VIT_CONFIG}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay, fused=True)

    print("\nStarting training...")

    # --- Training ---
    train_losses, val_losses, train_accus, val_accus = vit_training_eval_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        num_epoch=num_epochs,
        warmup_percent=warmup_percent,
        init_lr=init_lr,
        peak_lr=peak_lr,
        min_lr=min_lr,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        device=device,
        use_amp=amp,
    )

    print(f"Final training accuracy: {train_accus[-1]*100:.2f}%")
    print(f"Final validation accuracy: {val_accus[-1]*100:.2f}%")

    # --- Save model ---
    checkpoint_path = "vit_cifar10_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": TINY_VIT_CONFIG,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accus,
            "val_accuracies": val_accus,
        },
        checkpoint_path,
    )

    print(f"Model saved to {checkpoint_path}")
