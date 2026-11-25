
from pathlib import Path
import json
import numpy as np

try:
    from .load_data import load_raw, CHANNELS
except ImportError:
    from load_data import load_raw, CHANNELS


# Automatically detect the main project folder.
# This file should be inside the src folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Correct project paths.
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

VAL_SUBJECTS = [3, 15, 25, 30]

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main():

    print("Project folder:", PROJECT_ROOT)
    print("Dataset folder:", DATA_DIR)

    # Load raw dataset using the correct absolute path.
    raw = load_raw(DATA_DIR)

    train_raw = raw["train"]
    test_raw = raw["test"]

    # Check that all required validation volunteers exist.
    available_subjects = set(
        np.unique(train_raw["subject"]).tolist()
    )

    missing_subjects = sorted(
        set(VAL_SUBJECTS) - available_subjects
    )

    if missing_subjects:
        raise ValueError(
            f"Required validation subjects not found: {missing_subjects}"
        )

    # Split training data into training and validation.
    is_val = np.isin(train_raw["subject"], VAL_SUBJECTS)

    X_train_raw = train_raw["X"][~is_val]
    y_train = train_raw["y"][~is_val] - 1
    subject_train = train_raw["subject"][~is_val]

    X_val_raw = train_raw["X"][is_val]
    y_val = train_raw["y"][is_val] - 1
    subject_val = train_raw["subject"][is_val]

    # Keep the original test data separate.
    X_test_raw = test_raw["X"]
    y_test = test_raw["y"] - 1
    subject_test = test_raw["subject"]

    # Calculate mean and standard deviation using training data only.
    mean = X_train_raw.mean(axis=(0, 1))
    std = X_train_raw.std(axis=(0, 1))

    std[std == 0] = 1.0

    # Normalize training, validation, and testing data.
    X_train = ((X_train_raw - mean) / std).astype(np.float32)
    X_val = ((X_val_raw - mean) / std).astype(np.float32)
    X_test = ((X_test_raw - mean) / std).astype(np.float32)

    # Check that no volunteer occurs in multiple splits.
    train_people = set(np.unique(subject_train).tolist())
    val_people = set(np.unique(subject_val).tolist())
    test_people = set(np.unique(subject_test).tolist())

    assert train_people.isdisjoint(val_people)
    assert train_people.isdisjoint(test_people)
    assert val_people.isdisjoint(test_people)

    # Create folders if they do not exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed dataset.
    processed_path = DATA_DIR / "processed_data.npz"

    np.savez_compressed(
        processed_path,
        X_train=X_train,
        y_train=y_train.astype(np.int64),
        X_val=X_val,
        y_val=y_val.astype(np.int64),
        X_test=X_test,
        y_test=y_test.astype(np.int64),
        subject_train=subject_train,
        subject_val=subject_val,
        subject_test=subject_test,
    )

    # Save normalization values.
    scaler_path = DATA_DIR / "scaler.npz"

    np.savez(
        scaler_path,
        mean=mean,
        std=std
    )

    # Prepare dataset summary report.
    report = {
        "dataset_root": raw["dataset_root"],
        "channels": CHANNELS,
        "activities": ACTIVITY_NAMES,
        "validation_subjects": VAL_SUBJECTS,

        "train_windows": int(len(y_train)),
        "validation_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),

        "train_subjects": sorted(train_people),
        "validation_subjects_found": sorted(val_people),
        "test_subjects": sorted(test_people),

        "train_class_counts": np.bincount(
            y_train, minlength=6
        ).tolist(),

        "validation_class_counts": np.bincount(
            y_val, minlength=6
        ).tolist(),

        "test_class_counts": np.bincount(
            y_test, minlength=6
        ).tolist(),

        "training_channel_mean": mean.tolist(),
        "training_channel_std": std.tolist(),

        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
    }

    # Save JSON report.
    report_path = RESULTS_DIR / "data_summary.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    # Display original dataset information.
    print("\nRaw arrays:")

    for split in ["train", "test"]:

        d = raw[split]

        print(
            f"  {split}: X={d['X'].shape}, "
            f"y={d['y'].shape}, "
            f"subject={d['subject'].shape}"
        )

        print(
            "  Raw label counts (1..6):",
            np.bincount(d["y"], minlength=7)[1:].tolist()
        )

    # Display processed dataset information.
    print("\nAfter volunteer-based split:")

    for name, X, y, people in [
        ("train", X_train, y_train, train_people),
        ("validation", X_val, y_val, val_people),
        ("test", X_test, y_test, test_people),
    ]:

        print(
            f"  {name}: X={X.shape}, "
            f"y={y.shape}, "
            f"windows={len(y)}, "
            f"volunteers={sorted(people)}"
        )

        print(
            "    Class counts (labels 0..5):",
            np.bincount(y, minlength=6).tolist()
        )

    # Display normalization information.
    print("\nTraining-only normalization values:")

    for channel, channel_mean, channel_std in zip(
        CHANNELS, mean, std
    ):

        print(
            f"  {channel}: "
            f"mean={channel_mean:.8f}, "
            f"std={channel_std:.8f}"
        )

    # Display saved file locations.
    print("\nSuccessfully saved files:")

    print("Processed data:", processed_path)
    print("Scaler:", scaler_path)
    print("Summary report:", report_path)


if __name__ == "__main__":
    main()
