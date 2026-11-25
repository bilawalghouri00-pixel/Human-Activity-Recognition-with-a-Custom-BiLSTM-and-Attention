from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .load_data import load_raw, CHANNELS
except ImportError:
    from load_data import load_raw, CHANNELS

FS = 50


def plot_one_per_activity(split, names, out_path="results/figures/example_windows.png"):
    """Plot the first training window for each activity."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(128) / FS
    fig, axes = plt.subplots(2, 6, figsize=(21, 7), sharex=True)

    for class_index, name in enumerate(names):
        matches = np.flatnonzero(split["y"] == class_index + 1)
        if len(matches) == 0:
            continue
        window_index = int(matches[0])

        for channel_index in range(3):
            axes[0, class_index].plot(
                time_axis,
                split["X"][window_index, :, channel_index],
                label=CHANNELS[channel_index],
            )
            axes[1, class_index].plot(
                time_axis,
                split["X"][window_index, :, 3 + channel_index],
                label=CHANNELS[3 + channel_index],
            )

        axes[0, class_index].set_title(name.replace("_", " "), fontsize=9)
        axes[1, class_index].set_xlabel("Time (seconds)")

    axes[0, 0].set_ylabel("Body acceleration (g)")
    axes[1, 0].set_ylabel("Gyroscope (rad/s)")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    fig.suptitle("One raw sensor window for each activity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    raw = load_raw("data")
    plot_one_per_activity(raw["train"], raw["activity_names"])
