import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

sys.path.append("..") 

import random
import numpy as np
import pandas as pd

import generic_methods, DR_algorithms
import NMI

def measure_impact_of_missing_values(df_sample, algo):
	"""
    Computes the impact of missing values calls the methods from the NMI.py file

    Parameters
    ----------
    df_sample : pandas dataframe
        dataframe
    algo : string
        name of algorithm
    Returns
    ----------
    NMI.mutual_information : float
    	normalized mutual information score
    """
	data_matrix = df_sample.as_matrix()
	cols = list(df_sample.columns.values)
	
	for row_index in range(0,20):
	    row = random.randint(0, len(data_matrix)-1)
	    for col_index in range(0,5):
	        col = random.randint(0, len(data_matrix[1])-2)
	        data_matrix[row][col] = float(0)
	
	df_missing = pd.DataFrame(data = data_matrix, columns=cols)

	data_features, data_target = generic_methods.set_features_and_target(df_sample)
	embedding = DR_algorithms.run_DR_Algorithm(algo, data_features, data_target)

	missing_features, missing_target = generic_methods.set_features_and_target(df_missing)
	print(missing_features.shape)
	print(missing_target.shape)
	embedding_missing = DR_algorithms.run_DR_Algorithm(algo, missing_features, missing_target)

	return NMI.mutual_information((embedding, embedding_missing), k=10)

