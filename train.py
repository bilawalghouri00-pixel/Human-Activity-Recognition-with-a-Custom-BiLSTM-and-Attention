from pathlib import Path
import csv
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .model import HARNet, count_parameters
except ImportError:
    from model import HARNet, count_parameters

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
GRADIENT_CLIP = 1.0


def evaluate_loss_and_accuracy(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(X_batch)
            loss = loss_function(logits, y_batch)
            batch_count = len(y_batch)
            total_loss += loss.item() * batch_count
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_items += batch_count
    return total_loss / total_items, total_correct / total_items


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Two CPU threads are efficient for this small manually unrolled recurrent network.
    if device.type == "cpu":
        torch.set_num_threads(2)
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data = np.load("data/processed_data.npz")
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # The seed is set before the model is created, as required.
    torch.manual_seed(0)
    model = HARNet(input_size=9, hidden_size=64, num_classes=6).to(device)
    print("Trainable parameters:", count_parameters(model))
    if count_parameters(model) != 46982:
        raise RuntimeError("Parameter-count check failed. Expected exactly 46,982.")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial_train_loss, initial_train_accuracy = evaluate_loss_and_accuracy(
        model, train_loader, loss_function, device
    )
    initial_val_loss, initial_val_accuracy = evaluate_loss_and_accuracy(
        model, val_loader, loss_function, device
    )
    print(f"Initial training loss: {initial_train_loss:.6f} (log(6) = {np.log(6):.6f})")
    print(f"Initial validation loss: {initial_val_loss:.6f}")

    Path("results").mkdir(exist_ok=True)
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_items = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = loss_function(logits, y_batch)
            loss.backward()

            # Required: clip gradients after backward and before optimizer.step.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            batch_count = len(y_batch)
            total_train_loss += loss.item() * batch_count
            total_train_items += batch_count

        train_loss = total_train_loss / total_train_items
        val_loss, val_accuracy = evaluate_loss_and_accuracy(
            model, val_loader, loss_function, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val accuracy {val_accuracy * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "results/best_model.pt")

        # Save progress after every epoch so the history is not lost if interrupted.
        with open("results/training_history.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
            writer.writeheader()
            writer.writerows(history)

    training_seconds = time.time() - start_time

    with open("results/training_history.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    val_accuracies = [row["val_accuracy"] * 100 for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, train_losses, label="Training loss")
    axes[0].plot(epochs, val_losses, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accuracies, label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/training_curves.png", dpi=160)
    plt.close(fig)

    best_row = history[best_epoch - 1]
    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "trainable_parameters": count_parameters(model),
        "initial_training_loss": initial_train_loss,
        "initial_training_accuracy": initial_train_accuracy,
        "initial_validation_loss": initial_val_loss,
        "initial_validation_accuracy": initial_val_accuracy,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "training_loss_at_best_epoch": best_row["train_loss"],
        "validation_accuracy_at_best_epoch": best_row["val_accuracy"],
        "training_time_seconds": training_seconds,
        "epochs_completed": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP,
    }
    Path("results/training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 6))
    print("Saved model: results/best_model.pt")
    print("Saved curve: results/figures/training_curves.png")


if __name__ == "__main__":
    main()
