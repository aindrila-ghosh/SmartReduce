import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from context_local_structure import calculate_geodesic_distance
from sklearn.neighbors import NearestNeighbors

def run_n_points_ordering(df_np_order, points):
    '''
    ## In a loop, taking a sample of 3 points.
    ## Measuring distances among the chosen samples, before and after dimensionality reduction
    ## Creating an array with ranks of the distances, before and after dimensionality reduction
    ## Checking if the ranks are the same or not
    ## At the end, printing the number of times the ranks are the same (approximately 33% of the time with UMAP)
    '''

    results = []
    for times in range(0,1000):
        df_sample = df_np_order.sample(n=3)
        indexes_original = np.array(df_sample.index)
        embedding_sample = []

        for i in indexes_original:
            embedding_sample.append(points[i])

        dist1 = pdist(df_sample)
        dist2 = pdist(embedding_sample)

        temp1 = dist1.argsort()
        ranks1 = np.empty_like(temp1)
        ranks1[temp1] = np.arange(len(dist1))

        temp2 = dist2.argsort()
        ranks2 = np.empty_like(temp2)
        ranks2[temp2] = np.arange(len(dist2))

        results.append(np.array_equal(ranks1,ranks2))
    
    return (np.sum(results)/1000)

def run_K_max_ordering(features, points, k_max):
    """
    Computes K_max ordering of the data points

    Parameters
    ----------
    features : nD array
        original features
    points : nD array
        embedding 
    k_max : float
        value of k_max
    Returns
    ----------
    np.mean(mean_K_max_ordering) : float
        k_max ordering
    """
    if k_max <= 0:
        k_max = 1
    mean_K_max_ordering = []
    knn = NearestNeighbors(leaf_size=30, n_neighbors=k_max, p=2, radius=1.0, algorithm='ball_tree')
    knn_embd = NearestNeighbors(leaf_size=30, n_neighbors=k_max, p=2, radius=1.0, algorithm='ball_tree')
    knn.fit(features)
    knn_embd.fit(points)

    for index in range(0, len(features)):
        neighbors = knn.kneighbors(features, return_distance=False)[index]
        neighbors_embd = knn_embd.kneighbors(points, return_distance=False)[index]
        mean_K_max_ordering.append(len(np.intersect1d(neighbors, neighbors_embd))/len(neighbors))
        
    return np.mean(mean_K_max_ordering)