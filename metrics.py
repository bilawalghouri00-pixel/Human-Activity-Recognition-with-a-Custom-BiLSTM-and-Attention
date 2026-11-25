import numpy as np


def calculate_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def calculate_confusion_matrix(y_true, y_pred, num_classes=6):
    """Rows are actual classes and columns are predicted classes."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def calculate_per_class_accuracy(confusion_matrix):
    row_totals = confusion_matrix.sum(axis=1)
    correct = np.diag(confusion_matrix)
    return np.divide(
        correct,
        row_totals,
        out=np.zeros_like(correct, dtype=np.float64),
        where=row_totals != 0,
    )
