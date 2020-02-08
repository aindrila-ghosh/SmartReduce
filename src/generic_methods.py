import numpy as np
import pandas as pd

RANDOM_STATE = 42

def load_data(data, filetype):
    """
    loads data from CSV file into dataframe.
    Parameters
    ----------
    data : String
        path to input dataset
    filetype : String
        type of file
    Returns
    ----------
    df_Data : dataframe
        dataframe with data
    """

    if filetype == "csv":
        df_Data = pd.read_csv(data)
    else:
        df_Data = pd.read_excel(data)
    df_Data = df_Data.sample(1000)
    df_Data.reset_index(inplace = True)
    return df_Data

def set_features_and_target(df_Data):
    """
    Sets the features and targets from the dataframes.
    Parameters
    ----------
    df_Data : dataframe
        input dataset
    Returns
    ----------
    data_features : nD array
        matrix with data features
    data_target : list
        list of original labels
    """
    list_Data = list(df_Data)
    len_list = len(list_Data)
    features = list_Data[0:len(list_Data)-1]
    target = []
    target.append(list_Data[len(list_Data)-1])
    data_features = np.array(df_Data[features])
    data_target = np.array(df_Data[target])
    return data_features, data_target

 