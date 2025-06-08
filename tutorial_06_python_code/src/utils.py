import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def compute_mae(labels: pd.Series, predictions: pd.Series) -> float:
    """
    Compute MAE
    :param labels: given labels
    :param predictions: predicted values from a ML model

    return average MAE
    """
    mae = mean_absolute_error(labels, predictions)
    return mae

def compute_mape(labels: pd.Series, predictions: pd.Series) -> float:
    """
    Compute MAPE
    :param labels: given labels
    :param predictions: predicted values from a ML model

    return average MAPE
    """
    mape = mean_absolute_percentage_error(labels, predictions)
    return mape