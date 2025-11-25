from sklearn.metrics import mean_tweedie_deviance, auc
import numpy as np
import pandas as pd


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray,sample_weight: np.ndarray) -> float:
    """
    Compute the Gini coefficient using the weighted Lorenz curve.

    Parameters
    ----------
    y_true : np.ndarray
        True target values (e.g., PurePremium).
    y_pred : np.ndarray
        Model predictions for the target values.
    sample_weight : np.ndarray
        Exposure weights for each observation.

    Returns
    -------
    float
        The Gini coefficient (between -1 and 1),
        where higher values indicate better discriminatory power.
    """
    ranking = np.argsort(y_pred)
    ordered_weights = sample_weight[ranking]
    ordered_claims = (y_true * sample_weight)[ranking]
    cum_exposure = np.cumsum(ordered_weights) / ordered_weights.sum()
    cum_claims = np.cumsum(ordered_claims) / ordered_claims.sum()
    return 1 - 2 * auc(cum_exposure, cum_claims)


def evaluate_predictions(y_true: np.ndarray,y_pred: np.ndarray,sample_weight: np.ndarray | None = None) -> pd.DataFrame:
    """
    Compute core performance metrics for insurance claim models.

    Metrics include:
    - Bias (difference between weighted predicted and weighted actual means)
    - Tweedie deviance (power = 1.5)
    - Weighted MAE
    - Weighted RMSE
    - Gini coefficient (Lorenz-based)

    Parameters
    ----------
    y_true : np.ndarray
        True target values (e.g., PurePremium).
    y_pred : np.ndarray
        Model predictions for the target values.
    sample_weight : np.ndarray, optional
        Exposure weights for each observation. If None, unweighted
        metrics are computed.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame containing all evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_weight = np.ones_like(y_true) if sample_weight is None else np.asarray(sample_weight)


    # Bias
    actual_mean = (y_true * sample_weight).sum() / sample_weight.sum()
    pred_mean   = (y_pred * sample_weight).sum() / sample_weight.sum()
    bias = pred_mean - actual_mean

    # Deviance
    deviance = mean_tweedie_deviance(
        y_true,
        y_pred,
        power=1.5,
        sample_weight=sample_weight
        )

    # MAE
    mae = (sample_weight * np.abs(y_true - y_pred)).sum() / sample_weight.sum()

    #RMSE
    rmse = np.sqrt(
    ((sample_weight * (y_true - y_pred)**2).sum()) / sample_weight.sum()
    )

    #Gini
    gini = gini_coefficient(y_true, y_pred, sample_weight)

    #Return
    return pd.DataFrame(
    {
        "bias": [bias],
        "deviance": [deviance],
        "mae": [mae],
        "rmse": [rmse],
        "gini": [gini]
    }
)
