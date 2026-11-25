# Human Activity Recognition with a Custom BiLSTM and Attention

This project recognises six human activities from smartphone movement sensors. It follows the supplied assignment and uses the raw UCI HAR inertial-signal files.

## 1. What the project does

- Reads nine sensor signals for each 2.56-second window.
- Uses 128 time steps per window.
- Splits the original training volunteers into training and validation groups using validation volunteers `[3, 15, 25, 30]`.
- Calculates normalization mean and standard deviation from training windows only.
- Implements a simple RNN cell, LSTM cell, LSTM layer, Bidirectional LSTM, and temporal attention manually in PyTorch.
- Trains for 30 epochs with batch size 64, Adam learning rate 0.001, cross-entropy loss, and gradient clipping at 1.0.
- Tests the best validation-loss model once on the held-out test volunteers.
- Creates confusion matrix, per-class accuracy, attention plots, and a Streamlit application.

**Important:** The code does not use `nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.RNNCell`, `nn.LSTMCell`, `nn.GRUCell`, `nn.MultiheadAttention`, scikit-learn, TensorFlow, or Keras.

## 2. Project folders

```text
HAR_RNN_Assignment/
├── data/                      # Put and extract the UCI data here; do not upload to GitHub
├── src/
│   ├── load_data.py            # Read the nine original signal files
│   ├── plot_windows.py         # Draw one signal example for each activity
│   ├── prepare.py              # Split people, normalize, prepare tensors
│   ├── layers.py               # Custom RNN, LSTM, BiLSTM, Attention
│   ├── model.py                # Full HARNet model
│   ├── train.py                # Training loop
│   ├── metrics.py              # Accuracy and confusion-matrix functions
│   └── test.py                 # Final test and attention analysis
├── results/
│   ├── figures/                # Generated figures
│   ├── analysis.md             # Written task analysis
│   └── best_model.pt           # Saved after training; do not upload if too large
├── app/
│   └── app.py                  # Streamlit interface
├── blog/
│   └── medium_blog_draft.md    # Blog draft with result fields to complete
├── requirements.txt
└── README.md
```

## 3. Install Python

Use Python 3.11 or 3.12. Open Command Prompt or PowerShell in this project folder.

Check Python:

```bash
py --version
```

Create a virtual environment:

```bash
py -3.11 -m venv .venv
```

Activate it in **PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in that PowerShell window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

In **Command Prompt**, activate with:

```bat
.\.venv\Scripts\activate.bat
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Extract your uploaded dataset

Your uploaded ZIP has another ZIP inside it. Do both extraction steps.

1. Copy `human+activity+recognition+using+smartphones.zip` into the project folder.
2. Create the `data` folder if it does not exist.
3. In PowerShell, from the project folder, run:

```powershell
New-Item -ItemType Directory -Force data
Expand-Archive -Path ".\human+activity+recognition+using+smartphones.zip" -DestinationPath ".\data" -Force
Expand-Archive -Path ".\data\UCI HAR Dataset.zip" -DestinationPath ".\data" -Force
```

After extraction, this path must exist:

```text
data/UCI HAR Dataset/train/Inertial Signals/
```

It should contain files such as `body_acc_x_train.txt` and `body_gyro_z_train.txt`. The test folder should also contain nine files ending in `_test.txt`.

Do not use a Kaggle version that has only the 561 engineered features. This project needs the nine raw inertial signal files.

## 5. Run the assignment in the correct order

Run every command from the main `HAR_RNN_Assignment` folder, not from inside `src`.

### Step 1 — Check the raw dataset

```bash
python -m src.load_data
```

Expected main shapes:

```text
train X: (7352, 128, 9)
train y: (7352,)
test X:  (2947, 128, 9)
test y:  (2947,)
```

Expected raw activity counts:

| Activity | Train | Test |
|---|---:|---:|
| WALKING | 1,226 | 496 |
| WALKING_UPSTAIRS | 1,073 | 471 |
| WALKING_DOWNSTAIRS | 986 | 420 |
| SITTING | 1,286 | 491 |
| STANDING | 1,374 | 532 |
| LAYING | 1,407 | 537 |

### Step 2 — Split and normalize the data

```bash
python -m src.prepare
```

This command does all of these jobs:

1. Uses volunteers 3, 15, 25, and 30 for validation.
2. Keeps the remaining training volunteers in the training split.
3. Keeps the official test volunteers separate.
4. Changes labels from 1–6 to 0–5.
5. Calculates the nine means and standard deviations from training data only.
6. Applies those same values to validation and test data.
7. Saves `data/processed_data.npz`, `data/scaler.npz`, and `results/data_summary.json`.

Expected split sizes:

| Split | Windows | Volunteers |
|---|---:|---:|
| Training | 5,891 | 17 |
| Validation | 1,461 | 4 |
| Test | 2,947 | 9 |

### Step 3 — Create the first signal figure

```bash
python -m src.plot_windows
```

It creates:

`results/figures/example_windows.png`

Use this figure in your report and Medium blog.

### Step 4 — Check the custom network

```bash
python -m src.model
```

Check these outputs:

```text
LSTM output shape: (4, 128, 64)
BiLSTM output shape: (4, 128, 128)
Logits shape: (4, 6)
Attention shape: (4, 128)
Trainable parameters: 46982
```

The attention weights for each window should add up to approximately 1.0.

### Step 5 — Train the network

```bash
python -m src.train
```

The script runs all 30 epochs. It prints training loss, validation loss, and validation accuracy after every epoch. It saves the parameters from the epoch with the lowest validation loss to:

`results/best_model.pt`

It also saves:

- `results/training_history.csv`
- `results/training_summary.json`
- `results/results_tables.md` (fill its training-result cells after training/testing)
- `results/figures/training_curves.png`

Do not stop the script before it finishes all 30 epochs. The assignment says to use 30 epochs.

### Step 6 — Test the saved model

Only after training has finished, run:

```bash
python -m src.test
```

This loads the saved best model and tests every test window. It creates:

- `results/test_summary.json`
- `results/test_outputs.npz`
- `results/predictions.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/attention_examples.png`
- `results/figures/average_attention.png`

The confusion matrix must contain 2,947 total entries. Its rows are actual classes and its columns are predicted classes. Do not change the test set or use it during training.

### Step 7 — Start the web application

```bash
streamlit run app/app.py
```

The browser page lets you either choose a test window or upload your own CSV. Capture a screenshot of the running page and save it as `results/figures/app_screenshot.png`. Add it to the README using the screenshot line below. The CSV must have exactly 128 rows and 9 columns, without a header, using this exact column order:

```text
body_acc_x, body_acc_y, body_acc_z,
body_gyro_x, body_gyro_y, body_gyro_z,
total_acc_x, total_acc_y, total_acc_z
```

For a CSV upload, use raw sensor values. The application applies the saved training mean and standard deviation automatically.

Add the screenshot here after saving it in the path above:

```markdown
![HAR web application screenshot](results/figures/app_screenshot.png)
```

![HAR web application screenshot](results/figures/app_screenshot.png)

## 6. Parameter-count calculation

Let input size `D = 9` and hidden size `H = 64`.

| Layer | Calculation | Parameters |
|---|---:|---:|
| One LSTM cell | `4HD + 4H² + 4H` | 18,944 |
| Two directions | `2 × 18,944` | 37,888 |
| Attention | `64 × 128 + 64 + 64` | 8,320 |
| Output layer | `128 × 6 + 6` | 774 |
| **Total** | `37,888 + 8,320 + 774` | **46,982** |

The simple RNN cell is implemented for the assignment's comparison, but it is not part of the final HARNet, so its parameters are not counted in 46,982.

## 7. Important correctness checks

- LSTM returns `(B, 128, 64)`.
- BiLSTM returns `(B, 128, 128)`.
- Final logits return `(B, 6)`.
- Attention returns `(B, 128)`.
- Every attention value is positive and each window's 128 attention values sum to 1.
- The model has exactly 46,982 trainable parameters.
- The initial cross-entropy loss should be near `log(6) = 1.7918`.
- No volunteer is shared by training, validation, and test.
- The test confusion matrix sums to 2,947.
- The test set is not used to choose the best epoch.

The reference assignment reports test accuracy around 90–93%. This is an expected range, not a result to copy. Report the accuracy produced by your own run.

## 8. GitHub submission

Before uploading, check your files:

```bash
git status
```

Create the repository and commit your work in meaningful stages:

```bash
git init
git add README.md requirements.txt .gitignore src app blog results/analysis.md
git commit -m "Add HAR data preparation and custom recurrent layers"
```

After training and generating the figures, add your report figures and final documentation:

```bash
git add results/figures results/analysis.md blog
 git commit -m "Add training evaluation attention analysis and documentation"
```

Then create a public GitHub repository, connect it as `origin`, and push your branch. Do not add the original dataset, processed dataset, scaler, or large model file if your instructor does not ask for them. Keep meaningful commits from both group members; each member should make their own commits using their own GitHub account.

## 9. Final submission checklist

- [ ] Raw dataset extracted in the correct `data/` location.
- [ ] `python -m src.load_data` runs successfully.
- [ ] `python -m src.prepare` runs successfully.
- [ ] `example_windows.png` exists.
- [ ] Custom layer shapes and 46,982 parameter check pass.
- [ ] All 30 training epochs complete.
- [ ] Best validation-loss model is saved.
- [ ] Test confusion matrix sums to 2,947.
- [ ] Per-class accuracy is reported for all six activities.
- [ ] Six attention examples and six average attention curves are generated.
- [ ] `results/analysis.md` contains answers based on your actual results.
- [ ] Streamlit app runs and its screenshot is in the README.
- [ ] Medium blog is 800–1,500 words and contains at least two figures.
- [ ] LinkedIn post links the blog and public GitHub repository.
- [ ] Dataset is not committed to GitHub.
