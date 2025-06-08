import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing_extensions import Tuple

def split_data(data: pd.DataFrame, labels: pd.Series, test_size = 0.2, random_state=0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a given dataframe into train and test sets, with the corresponding labels

    :param data: data frame input
    :param labels: corresponding labels
    :param test_size: size of the test set in percentage. If none is given, default value is 0.2
    :param random_state: random state in integer. If none is given, default value is 0

    Return train, test data and their corresponding train, test labels
    """
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def impute_train_data(data: pd.DataFrame, columns: list[str], strategy="mean") -> pd.DataFrame:
    """
    Impute missing data of the train set, including the imputer

    :param data: data frame input
    :param columns: columns of data to be imputed
    :param strategy: strategy to create SimpleImputer. If none is given, default value is "mean" strategy

    return the used imputer, and the imputed dataframe
    """
    # create simple imputer with the given strategy input
    imputer = SimpleImputer(strategy=strategy)
    imputed_values = pd.DataFrame(
        imputer.fit_transform(data[columns]),
        index=data.index,
        columns=columns
    )
    return imputer, imputed_values

def impute_test_data(data: pd.DataFrame, columns: list[str], imputer: SimpleImputer) -> pd.DataFrame:
    """
    Impute missing data of the test set, given the imputer

    :param data: data frame input
    :param columns: columns of data to be imputed
    :param imputer: imputer that has been fitted on a train set

    return the imputed dataframe based on the given imputer
    """
    imputed_values = pd.DataFrame(
        imputer.transform(data[columns]),
        index=data.index,
        columns=columns
    )
    return imputed_values

def remove_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all categorical columns out of a given data frame

    :param data: data input

    Return a data frame with no categorical columns    
    """
    categorical_columns = data.select_dtypes(include="object").columns 
    new_data = data.drop(categorical_columns, axis=1)
    return new_data