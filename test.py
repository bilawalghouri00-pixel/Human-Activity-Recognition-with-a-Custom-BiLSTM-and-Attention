from pathlib import Path
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .model import HARNet
    from .metrics import (
        calculate_accuracy,
        calculate_confusion_matrix,
        calculate_per_class_accuracy,
    )
    from .load_data import CHANNELS
except ImportError:
    from model import HARNet
    from metrics import (
        calculate_accuracy,
        calculate_confusion_matrix,
        calculate_per_class_accuracy,
    )
    from load_data import CHANNELS

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def save_confusion_plot(matrix):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of windows")
    ax.set_xticks(range(6), [name.replace("_", " ") for name in ACTIVITY_NAMES], rotation=35, ha="right")
    ax.set_yticks(range(6), [name.replace("_", " ") for name in ACTIVITY_NAMES])
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("Actual activity")
    ax.set_title("Test confusion matrix")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(6):
        for col in range(6):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig("results/figures/confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_attention_examples(X_raw, y_true, y_pred, attention):
    time_axis = np.arange(128) / 50.0
    fig, axes = plt.subplots(2, 6, figsize=(22, 7), sharex=True)

    for class_id, name in enumerate(ACTIVITY_NAMES):
        candidates = np.flatnonzero((y_true == class_id) & (y_pred == class_id))
        if len(candidates) == 0:
            axes[0, class_id].text(0.5, 0.5, "No correctly classified example", ha="center", va="center")
            axes[1, class_id].axis("off")
            axes[0, class_id].set_title(name.replace("_", " "))
            continue

        chosen = int(candidates[0])
        for channel_id in range(3):
            axes[0, class_id].plot(time_axis, X_raw[chosen, :, channel_id], label=CHANNELS[channel_id])
        axes[0, class_id].set_ylabel("Body acc. (g)")
        axes[0, class_id].set_title(name.replace("_", " "))
        axes[0, class_id].legend(fontsize=6, loc="upper right")

        axes[1, class_id].plot(time_axis, attention[chosen])
        axes[1, class_id].set_ylabel("Attention")
        axes[1, class_id].set_xlabel("Time (s)")
    fig.suptitle("Correctly classified test examples and their attention", y=1.002)
    fig.tight_layout()
    fig.savefig("results/figures/attention_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_average_attention(y_true, attention):
    fig, ax = plt.subplots(figsize=(12, 6))
    mean_max_attention = {}
    for class_id, name in enumerate(ACTIVITY_NAMES):
        class_attention = attention[y_true == class_id]
        mean_curve = class_attention.mean(axis=0)
        ax.plot(np.arange(128) / 50.0, mean_curve, label=name.replace("_", " "))
        mean_max_attention[name] = float(class_attention.max(axis=1).mean())
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Average attention curve for each activity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/average_attention.png", dpi=160)
    plt.close(fig)
    return mean_max_attention


def main():
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    data = np.load("data/processed_data.npz")
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_true = data["y_test"].astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)
    model = HARNet(input_size=9, hidden_size=64, num_classes=6).to(device)
    state = torch.load("results/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_logits = []
    all_attention = []
    with torch.no_grad():
        for start in range(0, len(X_test), 64):
            batch = X_test[start:start + 64].to(device)
            logits, alpha = model(batch)
            all_logits.append(logits.cpu())
            all_attention.append(alpha.cpu())

    logits = torch.cat(all_logits, dim=0)
    attention = torch.cat(all_attention, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    y_pred = logits.argmax(dim=1).numpy()

    accuracy = calculate_accuracy(y_true, y_pred)
    confusion = calculate_confusion_matrix(y_true, y_pred, num_classes=6)
    per_class = calculate_per_class_accuracy(confusion)
    mean_max_attention = save_average_attention(y_true, attention)

    # Find the pair of different activities with the largest two-way confusion.
    confusion_pairs = []
    for first in range(6):
        for second in range(first + 1, 6):
            confusion_pairs.append((
                int(confusion[first, second] + confusion[second, first]),
                ACTIVITY_NAMES[first], ACTIVITY_NAMES[second]
            ))
    confusion_pairs.sort(reverse=True)
    most_confused_pair = confusion_pairs[0]
    moving_attention = float(np.mean([mean_max_attention[ACTIVITY_NAMES[i]] for i in [0, 1, 2]]))
    still_attention = float(np.mean([mean_max_attention[ACTIVITY_NAMES[i]] for i in [3, 4, 5]]))

    # Load unstandardized signals for easier-to-read plots in physical units.
    scaler = np.load("data/scaler.npz")
    X_test_raw = data["X_test"] * scaler["std"] + scaler["mean"]
    save_attention_examples(X_test_raw, y_true, y_pred, attention)
    save_confusion_plot(confusion)

    np.savez_compressed(
        "results/test_outputs.npz",
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        attention=attention,
    )

    with open("results/predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["window_index", "actual_activity", "predicted_activity"] +
                        [f"prob_{name.lower()}" for name in ACTIVITY_NAMES])
        for index in range(len(y_true)):
            writer.writerow([index, ACTIVITY_NAMES[y_true[index]], ACTIVITY_NAMES[y_pred[index]], *probabilities[index].tolist()])

    report = {
        "test_accuracy": accuracy,
        "test_accuracy_percent": accuracy * 100,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": {
            ACTIVITY_NAMES[i]: float(per_class[i]) for i in range(6)
        },
        "mean_max_attention": mean_max_attention,
        "most_confused_pair": {
            "activity_1": most_confused_pair[1],
            "activity_2": most_confused_pair[2],
            "two_way_error_count": most_confused_pair[0],
        },
        "moving_mean_max_attention": moving_attention,
        "still_mean_max_attention": still_attention,
        "attention_min": float(attention.min()),
        "attention_max": float(attention.max()),
        "attention_row_sum_min": float(attention.sum(axis=1).min()),
        "attention_row_sum_max": float(attention.sum(axis=1).max()),
        "confusion_matrix_total": int(confusion.sum()),
    }
    Path("results/test_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Test accuracy: {:.2f}%".format(accuracy * 100))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion)
    print("Per-class accuracy:")
    for name, value in report["per_class_accuracy"].items():
        print(f"  {name}: {value * 100:.2f}%")
    print("Mean of max attention:")
    for name, value in mean_max_attention.items():
        print(f"  {name}: {value:.6f}")
    print("Most confused pair:", most_confused_pair[1], "and", most_confused_pair[2],
          "(two-way errors:", most_confused_pair[0], ")")
    print(f"Moving average peak attention: {moving_attention:.6f}")
    print(f"Still average peak attention: {still_attention:.6f}")
    print("Confusion matrix total:", confusion.sum(), "(must be 2947)")
    print("Attention sum range:", report["attention_row_sum_min"], report["attention_row_sum_max"])
    print("Saved test reports, predictions and figures in results/.")


if __name__ == "__main__":
    main()
