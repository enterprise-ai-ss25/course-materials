import pandas as pd 
from sklearn.ensemble import RandomForestRegressor

def create_forest_regressor(random_state=0) -> RandomForestRegressor:
    """
    Create a machine learning model given a random state
    
    :param random_state: random_state to generate the model. If none is given, the default value is 0.

    Return a RandomForestRegressor model with the given random state
    """
    model = RandomForestRegressor(random_state=random_state)
    return model 

def train_model(model: RandomForestRegressor, X_train: pd.DataFrame, Y_train: pd.Series) -> RandomForestRegressor:
    """
    Train a given model

    :param model: model input
    :param X_train: train data input
    :param Y_train: train label input

    Return a trained model
    """
    model.fit(X_train, Y_train)
    return model

def get_predictions(model: RandomForestRegressor, data: pd.DataFrame) -> pd.Series:
    """
    Get predicted values from a trained model

    :param model: model input
    :param data: data input

    Return a pandas Series of predictions
    """
    predictions = model.predict(data)
    return predictions