# Task 6 — Analysis

Complete the result-dependent sentences after running `python -m src.test`. Use the actual values from `results/test_summary.json`. Do not invent results.

## 6.1 Why is validation split by person?

The dataset has windows that overlap by 50%. This means two nearby windows can contain many of the same sensor readings. If random windows were divided into training and validation, very similar windows from the same person could appear in both groups. The model could then appear to perform well because it has already seen very similar movement patterns. Splitting by volunteer gives a fairer test of how the model works for people it did not train on.

## 6.2 Which two activities are confused most often?

Read `most_confused_pair` in `results/test_summary.json`. The assignment notes that SITTING and STANDING are commonly confused. In the final run, the pair with the largest combined two-way confusion is **[ADD THE TWO ACTIVITIES FROM YOUR JSON]**, with **[ADD COUNT]** errors in both directions combined.

A phone is attached at the waist, so sitting and standing can produce similar sensor patterns. In both activities the person is mostly still, and the phone may have a similar orientation. Their movement signals can therefore look alike in a short 2.56-second window.

## 6.3 Why is LAYING usually recognised well?

The three total-acceleration signals (`total_acc_x`, `total_acc_y`, and `total_acc_z`) include gravity. When a person lies down, the phone's orientation relative to gravity is different from when the person is upright. This creates a useful and relatively stable pattern in the total-acceleration channels. The body-acceleration channels have gravity removed, so the total-acceleration channels are especially useful for identifying body orientation.

Check the LAYING row in the confusion matrix and report the actual accuracy from `test_summary.json`.

## 6.4 Is attention more peaked for moving or still activities?

The test script calculates the average of each activity's maximum attention value. It then calculates the average for the three moving activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS) and the three still activities (SITTING, STANDING, LAYING).

From `results/test_summary.json`, moving average peak attention is **[ADD VALUE]** and still average peak attention is **[ADD VALUE]**. The larger value indicates the group for which attention is more concentrated around a smaller number of time steps. A smaller value means attention is spread more evenly across the window. Use the generated `average_attention.png` to discuss the actual shape of the curves too.

## 6.5 Why is a bidirectional LSTM allowed here?

The model receives a complete 2.56-second window before making a prediction. It already has all 128 time steps, including the later part of the window. Therefore, it is allowed to read the same window forward and backward. This would not be suitable for a task that must predict the next event in real time before future sensor readings are available, such as predicting the next movement at the current instant.

## 6.6 Why does an LSTM keep information better, and what does gradient clipping do?

An LSTM has a cell state that carries information through time. Its update is `c_t = f_t × c_(t-1) + i_t × g_t`. The forget gate decides how much old information to keep, while the input gate decides how much new information to add. This controlled path helps the LSTM keep useful information over many time steps. A plain RNN repeatedly applies a tanh update, which can make information and gradients become very small during long sequences.

Gradient clipping limits the total gradient size before the optimizer updates the weights. It helps prevent very large gradients from causing unstable or excessively large updates. In this assignment, the maximum gradient norm is 1.0.

## Result values to copy into this file

- Test accuracy: **[FROM test_summary.json]**
- Most confused pair: **[FROM test_summary.json]**
- Moving average peak attention: **[FROM test_summary.json]**
- Still average peak attention: **[FROM test_summary.json]**
- LAYING per-class accuracy: **[FROM test_summary.json]**
