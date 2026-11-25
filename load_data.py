
from pathlib import Path
import numpy as np

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def find_dataset_root(data_dir="data"):
    """Find the actual UCI HAR Dataset folder and ignore __MACOSX."""

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder does not exist: {data_dir.resolve()}"
        )

    for signal_folder in sorted(data_dir.rglob("Inertial Signals")):

        # Ignore unwanted macOS folders
        if "__MACOSX" in signal_folder.parts:
            continue

        # The Inertial Signals folder must be inside train
        if signal_folder.parent.name != "train":
            continue

        root = signal_folder.parent.parent

        # Check that this is the actual dataset
        required_files = [
            root / "activity_labels.txt",
            root / "train" / "y_train.txt",
            root / "train" / "subject_train.txt",
            root / "test" / "y_test.txt",
            root / "test" / "subject_test.txt",
        ]

        for split in ["train", "test"]:
            for channel in CHANNELS:
                required_files.append(
                    root / split / "Inertial Signals" /
                    f"{channel}_{split}.txt"
                )

        if all(file.is_file() for file in required_files):
            return root

    raise FileNotFoundError(
        f"Actual UCI HAR Dataset not found inside: {data_dir.resolve()}\n"
        "Make sure the dataset is extracted correctly and contains "
        "train/, test/, and Inertial Signals/ folders."
    )


def load_raw(data_dir="data"):
    """Load the nine raw sensor channels. Labels remain in the range 1..6."""

    root = find_dataset_root(data_dir)
    output = {}

    for split in ["train", "test"]:

        signal_folder = root / split / "Inertial Signals"
        channels = []

        for channel in CHANNELS:
            file_path = signal_folder / f"{channel}_{split}.txt"

            values = np.loadtxt(file_path, dtype=np.float32)
            channels.append(values)

        X = np.stack(channels, axis=-1)

        y = np.loadtxt(
            root / split / f"y_{split}.txt",
            dtype=np.int64
        )

        subject = np.loadtxt(
            root / split / f"subject_{split}.txt",
            dtype=np.int64
        )

        output[split] = {
            "X": X,
            "y": y,
            "subject": subject
        }

    activity_names = []

    with (root / "activity_labels.txt").open(
        "r", encoding="utf-8"
    ) as file:

        for line in file:
            if line.strip():
                activity_names.append(
                    line.split(maxsplit=1)[1].strip()
                )

    output["activity_names"] = activity_names
    output["dataset_root"] = str(root)

    return output


if __name__ == "__main__":

    raw = load_raw("data")

    print("Dataset folder:", raw["dataset_root"])

    for split in ["train", "test"]:

        data = raw[split]

        print(
            f"{split}: X={data['X'].shape}, "
            f"y={data['y'].shape}, "
            f"subject={data['subject'].shape}"
        )

        print(
            "  volunteers:",
            sorted(np.unique(data["subject"]).tolist())
        )

    print("Activities:", raw["activity_names"])
    print("Channel order:", CHANNELS)
