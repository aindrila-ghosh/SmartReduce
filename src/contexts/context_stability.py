import os
import sys
sys.path.append("..") 

import pandas as pd
import numpy as np
import DR_algorithms
import generic_methods
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


"""
This code checks for the Reproducibility of DR algorithms with Partial Records
"""


def custom_range(start, stop, step):
    """
    Generates a custom range

    Parameters
    ----------
    start : integer
        start value of range
    stop : integer
        stop value of range
    step : integer
        step value of range
    Returns
    ----------
    nothing!
    """
    numelements = int((stop-start)/float(step))
    for i in range(numelements+1):
            yield start + i*step

def run_partial_column_experiments(algo, df_Original_Data, original_embedding):
    """
    Runs partial column experiments on the datasets. That is the code removes one column at a time and checks the 
    structural difference of the purturbed embedding with the complete embedding

    Parameters
    ----------
    algo : string
        name of algorithm
    df_Original_Data : nD array
        original dataset
    original_embedding: nD array
        embedding
    Returns
    ----------
    np.mean(differences) : float
        the Structural Similarity Index between the original and the purturbed embeddings
    """
    df_Data = df_Original_Data
    
  # drop columns one by one from the dataset

    differences = []

    for x in range(0,len(df_Original_Data.columns)-1):
        
        print("Column...... ", x)
        df_Data_Subset = df_Data.drop(df_Data.columns[x], axis=1)
        data_features, data_target = generic_methods.set_features_and_target(df_Data_Subset)
        embedding = DR_algorithms.run_DR_Algorithm(algo, data_features, data_target)

        ## calculate structural differences
        difference = np.ceil(np.sum((original_embedding-embedding)**2)/1000.)
        differences.append(difference)

        df_Data = df_Original_Data

    return np.mean(differences)


def run_partial_row_experiments(algo, df_Original_Data, original_embedding):
    """
    Runs partial row experiments on the datasets. That is the code takes horizontal subsets of the original data and checks the 
    logarithmic loss of the purturbed embedding with respect to the complete embedding

    Parameters
    ----------
    algo : string
        name of algorithm
    df_Original_Data : nD array
        original dataset
    original_embedding: nD array
        embedding
    Returns
    ----------
    np.mean(losses) : float
        the kNN logarithmic loss of the purturbed embedding
    """

    df_Data = df_Original_Data
    int_dim =2
    #df_Data = df_Data.reset_index()

    losses = []

    if algo == "LTSA" or algo == "MDS" or algo == "MVU" or algo == "HessianLLE" or algo == "LLE" or algo == "MLLE":
        for i in custom_range(50, 25, 100):

            df_Data = df_Data.sample(n=i)
            data_features, data_target = generic_methods.set_features_and_target(df_Data)
            embedding = DR_algorithms.run_DR_Algorithm(algo, data_features, data_target, int_dim)

            loss = 0
            ## calculate log-loss
            train_features, test_features, train_labels, test_labels = train_test_split(embedding, data_target, test_size = 0.25, random_state = 42)
            knn = KNeighborsClassifier(3)
            knn.fit(train_features, train_labels.ravel())
            predictions_knn = knn.predict(test_features)
            try:
            	loss = log_loss(predictions_knn, test_labels)
            except ValueError:
            	pass         

            losses.append(loss)
            
            df_Data = df_Original_Data
    else:
        for i in custom_range(5000, 2500, 1000):

            df_Data = df_Data.sample(n=i)
            data_features, data_target = generic_methods.set_features_and_target(df_Data)
            embedding = DR_algorithms.run_DR_Algorithm(algo, data_features, data_target, int_dim)

            ## calculate log-loss
            train_features, test_features, train_labels, test_labels = train_test_split(embedding, data_target, test_size = 0.25, random_state = 42)

            knn = KNeighborsClassifier(3)
            knn.fit(train_features, train_labels.ravel())
            predictions_knn = knn.predict(test_features)
            loss = log_loss(predictions_knn, test_labels)

            losses.append(loss)
            
            df_Data = df_Original_Data

    return np.mean(losses)




