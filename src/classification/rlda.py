"""
Shrinkage LDA (rLDA) on trial features.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_rlda(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit z-scoring and LDA with analytic Ledoit-Wolf shrinkage on the training trials
    and predict the class of every test trial. Any number of classes.

    Parameters
    ----------
    X_train : np.ndarray
        Dims (trial, feature).
    y_train : np.ndarray
        The class of each training trial.
    X_test : np.ndarray
        Dims (trial, feature).

    Returns
    -------
    np.ndarray
        The predicted class of each test trial.
    """
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return clf.fit(X_train, y_train).predict(X_test)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray,
                     classes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """True class to predicted class to number of trials, every class listed."""
    counts = confusion_matrix(y_true, y_pred, labels=list(classes))
    return {true: {pred: int(counts[i, j]) for j, pred in enumerate(classes)}
            for i, true in enumerate(classes)}
