import numpy as np
import pandas as pd

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri


def set_envrionment():

    """
    Setting rPy2 environmets
    """

    base = rpackages.importr('base')
    packnames = ('dimRed')

    if all(rpackages.isinstalled(x) for x in packnames):
	    installed = True
    else:
	    utils = rpackages.importr('utils')
	    utils.chooseCRANmirror(ind=1) 
	    utils.install_packages(StrVector(packnames))

    return dimRed

def calculate_dimRed_metrics(points, df_sample):

    """
    Computes metrics based on correlation matrix

    Parameters
    ----------
    points : nD array
        embedding
    df_sample : nD array
        original data
    Returns
    ----------
    AUC_log_R_NX : float
    mean_R_NX : float
    Q_local : float
    Q_global : float
    K_max : integer
    """

	#dimRed = set_environment()

    rpy2.robjects.numpy2ri.activate()
    pandas2ri.activate()
    dimRed = rpackages.importr('dimRed')

    nr,nc = points.shape
    Pointsr = robjects.r.matrix(points, nrow=nr, ncol=nc)
    robjects.r.assign("points", Pointsr)

    nr,nc = df_sample.as_matrix().shape
    df_sample_matrix = robjects.r.matrix(df_sample.as_matrix(), nrow=nr, ncol=nc)
    robjects.r.assign("df_sample.as_matrix()", df_sample_matrix)
    
    drObject = dimRed.dimRedData(data=Pointsr, meta=df_sample)

    params = {'data': drObject, 'org.data': df_sample_matrix, 'has.org.data' : True}
    drResultObject = dimRed.dimRedResult(**params)

    AUC_log_R_NX = float(dimRed.AUC_lnK_R_NX(drResultObject))
    mean_R_NX = dimRed.mean_R_NX(drResultObject)
    Q_local = dimRed.Q_local(drResultObject)
    Q_global = dimRed.Q_global(drResultObject)
    K_max = np.argmax(dimRed.LCMC(drResultObject))
    
    return AUC_log_R_NX, mean_R_NX, Q_local, Q_global, K_max