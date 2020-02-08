from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def intrinsic_dim_sample_wise(X, k=5):

    """
    Helper function: Computes the intrinsic dimensionality of a dataset.

    """
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):

    """
    Helper function: Computes the intrinsic dimensionality of a dataset.
    """
    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):

    """
    Helper function: Computes the intrinsic dimensionality of a dataset.
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)    
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results

def compute_intrinsic_dim(datasets):

    """
    Computes the intrinsic dimensionality of a dataset.
    Parameters
    ----------
    datasets : tuple
        tuple of datasets
    Returns
    ----------
    intrinsic_dimensionality : array
        array of intrinsic dimensionality of each dataset
    """

    intrinsic_dimensionality = []

    for index_dataset, (dataset_name, df_data) in enumerate(datasets):
        intdim_k_repeated = Preprocessing.repeated(Preprocessing.intrinsic_dim_scale_interval, 
                                 df_data.as_matrix(), 
                                 mode='bootstrap', 
                                 nb_iter=500, # nb_iter for bootstrapping
                                 verbose=1, 
                                 k1=10, k2=20) # start and end of interval
        intdim_k_repeated = np.array(intdim_k_repeated)
        
        histogram = np.histogram(intdim_k_repeated.mean(axis=1))
        int_dim_original = histogram[1][np.argsort(histogram[0])[len(histogram[0])-1]]
        if int_dim_original - math.floor(int_dim_original) < 0.5:
            int_dim = math.floor(int_dim_original)
        else:
            int_dim = math.ceil(int_dim_original)
        intrinsic_dimensionality.append(int_dim)

    return intrinsic_dimensionality