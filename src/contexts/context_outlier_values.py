
import pandas as pd
import numpy as np
import ssim

def mad(a, axis=None):

    """
    Compute Median Absolute Deviation of an array along given axis.
    Median along given axis, but *keeping* the reduced axis so that result can still broadcast against a.

    Parameters
    ----------
    a : array
        input array
    Returns
    ----------
    mad : float
        median absolute deviation
    """
    
    med = np.median(a, axis=axis, keepdims=True)
    mad = np.median(np.absolute(a - med), axis=axis)  # MAD along given axis
    return mad

def convert_df_to_array(df_Data, column):
    """
    Converts data frames to arrays

    Parameters
    ----------
    df_Data : pandas dataframe
        input dataframe
    columns : list
        columns of the input dataframe
    Returns
    ----------
    array_val : array
        array version of the dataframe
    """
    array_val = df_Data[column].values
    return array_val

def check_for_outlier_values(column, value):
    """
    Checks for outlier values in a column

    Parameters
    ----------
    column : list
        input column
    value : float
        outlier
    Returns
    ----------
    outliers : array
        array of outlier values
    """
    count = 0
    outliers = []
    for i in range(0, len(column)):
        if (abs(column[i] - np.median(column)) / mad(column)) > value:
            outliers.append (column[i])
            count = count + 1
    #print(count)
    if len(outliers) == 0:
    	#print("no outliers in the column: ", column)
    	outlier = (value * mad(column)) + np.median(column)
    	outliers.append(outlier)

    return outliers


def measure_difference_between_embeddingds(algo, original_embedding, outlier_embedding):
    """
    Measures the differences between the original embedding and the embedding with outliers using Structural Similarity Index

    Parameters
    ----------
    original_embedding : nD array
        original embedding
    outlier_embedding : nD array
        embedding with outliers (same dimension as the original embedding)
    Returns
    ----------
    difference : float
        the Structural Similarity Index between the original and the otlier infused embeddings
    """

    embedding = np.array(outlier_embedding).astype(np.float64)
    ref_embedding = np.array(original_embedding).astype(np.float64)

    mssim, ssim = compute_ssim(embedding, ref_embedding)

    
    return mssim