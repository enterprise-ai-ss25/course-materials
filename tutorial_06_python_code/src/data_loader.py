import pandas as pd 

def load_data(data_path) -> pd.DataFrame:
    """ Load dataset function
    :param data_path: absolute path of the dataset, it could also be an online UR

    return a dataset in the Pandas DataFrame format
    """
    data = pd.read_csv(data_path)
    return data