import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_weights():
    return np.array([[1.0, -0.5, -1.0], [0.1, 1.0, 0.0], [0.0, 0.25, 1.0]])


def three_class_solubility(y_true, y_pred, sample_weight=None, **kwargs):
    # For balanced accuracy, with W = I: np.diag(C) = np.sum(C * W)
    # In MCC, W = 2 * I - 1 (ie. off diagonals are -1 instead of 0)
    W = get_confusion_weights()
    try:
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    except UserWarning:
        print("True, Predicted, and Confusion Weighting")
        print(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(C), np.int16))
            )
        )
        C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # with np.errstate(divide="ignore", invalid="ignore"):
    with np.errstate(divide="print", invalid="print"):
        per_class = np.sum(C * W, axis=1) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        raise UserWarning.add_note(
            "\ny_pred contains classes not in y_true:\n{}\n".format(
                np.argwhere(np.astype(np.isnan(per_class), np.int16))
            )
        )
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score
