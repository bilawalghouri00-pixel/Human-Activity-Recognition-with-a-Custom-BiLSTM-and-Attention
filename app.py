from pathlib import Path
import io
import sys
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import HARNet

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


@st.cache_resource
def load_model():
    model = HARNet(input_size=9, hidden_size=64, num_classes=6)
    weights_path = PROJECT_ROOT / "results" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_test_data():
    data = np.load(PROJECT_ROOT / "data" / "processed_data.npz")
    scaler = np.load(PROJECT_ROOT / "data" / "scaler.npz")
    return data["X_test"], data["y_test"], scaler["mean"], scaler["std"]


def main():
    st.set_page_config(page_title="HAR Activity Recognition", layout="wide")
    st.title("Human Activity Recognition")
    st.write("Choose a saved UCI test window or upload your own CSV file with exactly 128 rows and 9 columns.")

    model_path = PROJECT_ROOT / "results" / "best_model.pt"
    processed_path = PROJECT_ROOT / "data" / "processed_data.npz"
    scaler_path = PROJECT_ROOT / "data" / "scaler.npz"
    if not (model_path.exists() and processed_path.exists() and scaler_path.exists()):
        st.error("Required files are missing. First run prepare.py, train.py and test.py from the project root.")
        st.stop()

    model = load_model()
    X_test, y_test, mean, std = load_test_data()
    method = st.radio("Choose input", ["Choose a test window", "Upload a CSV file"], horizontal=True)

    if method == "Choose a test window":
        index = st.number_input("Test window number", min_value=0, max_value=len(X_test) - 1, value=0, step=1)
        original = X_test[int(index)] * std + mean
        actual_label = ACTIVITY_NAMES[int(y_test[int(index)])]
        st.info(f"Actual activity in this test window: {actual_label}")
        model_input = X_test[int(index)]
    else:
        uploaded = st.file_uploader("Upload CSV (128 rows × 9 columns, no header)", type=["csv"])
        if uploaded is None:
            st.stop()
        try:
            original = np.loadtxt(io.BytesIO(uploaded.getvalue()), delimiter=",")
        except Exception as error:
            st.error(f"Could not read this CSV: {error}")
            st.stop()
        if original.shape != (128, 9):
            st.error(f"Your file has shape {original.shape}. It must have exactly 128 rows and 9 columns.")
            st.stop()
        model_input = ((original - mean) / std).astype(np.float32)
        actual_label = "Not provided for uploaded data"

    with torch.no_grad():
        input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
        logits, alpha = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
        predicted_id = int(probabilities.argmax())
        attention = alpha[0].numpy()

    st.subheader("Prediction")
    st.success(f"Predicted activity: {ACTIVITY_NAMES[predicted_id]}")
    st.write("Actual activity:", actual_label)

    left, right = st.columns(2)
    with left:
        st.subheader("Probabilities of all six activities")
        probability_figure, probability_axis = plt.subplots(figsize=(8, 4))
        probability_axis.bar([name.replace("_", " ") for name in ACTIVITY_NAMES], probabilities)
        probability_axis.set_ylim(0, 1)
        probability_axis.set_ylabel("Probability")
        probability_axis.tick_params(axis="x", rotation=35)
        probability_figure.tight_layout()
        st.pyplot(probability_figure)
        plt.close(probability_figure)

    with right:
        st.subheader("Attention over the 2.56-second window")
        attention_figure, attention_axis = plt.subplots(figsize=(8, 4))
        attention_axis.plot(np.arange(128) / 50.0, attention)
        attention_axis.set_xlabel("Time (seconds)")
        attention_axis.set_ylabel("Attention weight")
        attention_figure.tight_layout()
        st.pyplot(attention_figure)
        plt.close(attention_figure)

    st.subheader("Nine sensor signals")
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    time_axis = np.arange(128) / 50.0
    groups = [(0, 3, "Body acceleration"), (3, 6, "Gyroscope"), (6, 9, "Total acceleration")]
    for axis, (start, end, title) in zip(axes, groups):
        for channel_index in range(start, end):
            axis.plot(time_axis, original[:, channel_index], label=CHANNELS[channel_index])
        axis.set_title(title)
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.caption("For an uploaded CSV, columns must be in this order: " + ", ".join(CHANNELS))


if __name__ == "__main__":
    main()
