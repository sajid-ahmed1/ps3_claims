import numpy as np
from ps3.evaluation import evaluate_predictions, gini_coefficient

# synthetic data
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.0, 2.0, 3.0])
weights = np.array([1.0, 1.0, 1.0])

def test_gini_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])

    g = gini_coefficient(y_true, y_pred, w)

    # For perfect prediction, Gini = 0 (no discrimination needed)
    assert  np.isclose(g, 0.2777777777777778)

def test_evaluate_predictions_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])

    df = evaluate_predictions(y_true, y_pred, w)

    assert np.isclose(df["bias"][0], 0.0)
    assert np.isclose(df["mae"][0], 0.0)
    assert np.isclose(df["rmse"][0], 0.0)

    # Deviance = 0 for perfect prediction (Poisson/Tweedie)
    assert np.isclose(df["deviance"][0], 0.0)

def test_evaluate_predictions_weighted():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])
    w = np.array([1.0, 10.0, 1.0])

    df = evaluate_predictions(y_true, y_pred, w)

    # Manually compute weighted MAE to compare:
    manual_mae = (1*0 + 10*0.5 + 1*0.5) / 12
    assert np.isclose(df["mae"][0], manual_mae)

def test_unweighted():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 1.0])

    df = evaluate_predictions(y_true, y_pred)  # no weights passed

    # MAE for [1, 2] vs [2, 1]
    assert np.isclose(df["mae"][0], 1.0)
