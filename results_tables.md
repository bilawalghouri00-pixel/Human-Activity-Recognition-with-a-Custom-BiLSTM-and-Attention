# Results tables

## Table 1 — Data

| Split | Windows | Volunteers |
|---|---:|---:|
| Training | 5891 | 17 |
| Validation | 1461 | 4 |
| Test | 2947 | 9 |

### Training channel means and standard deviations

| Channel | Training mean | Training standard deviation |
|---|---:|---:|
| body_acc_x | -0.00070241 | 0.19890375 |
| body_acc_y | -0.00026787 | 0.12258008 |
| body_acc_z | -0.00032155 | 0.10899112 |
| body_gyro_x | -0.00075935 | 0.40147191 |
| body_gyro_y | -0.00076526 | 0.38922656 |
| body_gyro_z | -0.00029247 | 0.26512876 |
| total_acc_x | 0.79996979 | 0.41944245 |
| total_acc_y | 0.02070495 | 0.38310423 |
| total_acc_z | 0.07646593 | 0.37672880 |

## Table 2 — Network and training

| Item | Your run result |
|---|---|
| Trainable parameters | 46,982 |
| Initial training loss | [COPY FROM results/training_summary.json] |
| Best epoch | [COPY FROM results/training_summary.json] |
| Training loss at best epoch | [COPY FROM results/training_summary.json] |
| Validation loss at best epoch | [COPY FROM results/training_summary.json] |
| Validation accuracy at best epoch | [COPY FROM results/training_summary.json] |
| Training time and hardware | [COPY FROM results/training_summary.json] |

## Table 3 — Test results

| Metric | WALK | UPSTAIRS | DOWNSTAIRS | SITTING | STANDING | LAYING |
|---|---:|---:|---:|---:|---:|---:|
| Per-class accuracy | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] |
| Mean of maximum attention | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] | [RUN TEST] |

**Overall test accuracy:** [COPY FROM results/test_summary.json]

**Confusion matrix:** use `results/test_summary.json` or `results/figures/confusion_matrix.png`. Rows are actual activity names; columns are predicted activity names.
