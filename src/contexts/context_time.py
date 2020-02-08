import time
import seaborn as sns
import umap, trimap, time
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding, SpectralEmbedding, locally_linear_embedding
RANDOM_STATE = 42

def plot_scatter(points, data_target):

    """
    Creates 2D scatter plots of the embeddings.
    Parameters
    ----------
    points : nD array
        embedding
    data_target : list
        original labels
    """

    sns.set_style("darkgrid")
    sns.scatterplot(points[:,0], points[:,1], hue=data_target.ravel())
    plt.title("low dimensional embedding", fontsize=20, y=1.03)
    plt.xlabel("1st component", fontsize=16)
    plt.ylabel("2nd component", fontsize=16)
    plt.show()

def timer(start,end):
    """
    Computes the time difference between start and end.
    Parameters
    ----------
    start : time
        start time
    end : time
        end time
    """

    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)


def run_DR_Algorithm(name, data_features, data_target, int_dim=2):
    """
    Runs each DR algorithm and computes the time duration.
    Parameters
    ----------
    name : String
        start time
    data_features : nD array
        original features
    data_target : list
        original labels
    int_dim : integer
        intrinsic dimensionality
    Returns
    ----------
    end-start: float
        duration of execution
    """

    start = time.time()

    if name == "UMAP":
        reducer = umap.UMAP()
        points = reducer.fit_transform(data_features)

    elif name == "tSNE":
        tsne = TSNE(n_components=int_dim, n_iter=1000, random_state=RANDOM_STATE)
        points = tsne.fit_transform(data_features)
    
    elif name == "PCA":
        pca = PCA(n_components=int_dim)
        points = pca.fit_transform(data_features)

    elif name == "FIt_SNE":
        points = fast_tsne(data_features, perplexity=50, seed=42)

    elif name == "M_Core_tSNE":
        tsne = M_TSNE(n_jobs=4)
        points = tsne.fit_transform(data_features)

    elif name == "Trimap":
        points = trimap.TRIMAP().fit_transform(data_features)

    elif name == "dPCA":
        dpca = dPCA.dPCA(labels='st', n_components=int_dim)
        points = dpca.fit_transform(data_features)

    elif name == "LTSA":
        ltsa = LocallyLinearEmbedding(n_neighbors = 12, n_components=int_dim, method = 'ltsa')
        points = ltsa.fit_transform(data_features)

    elif name == "MLLE":
        ltsa = LocallyLinearEmbedding(n_neighbors = 6, n_components=int_dim, method = 'modified')
        points = ltsa.fit_transform(data_features)

    elif name == "openTSNE":
        tsne = opTSNE(
                n_components=int_dim, perplexity=30, learning_rate=200,
                n_jobs=4, initialization="pca", metric="euclidean",
                early_exaggeration_iter=250, early_exaggeration=12, n_iter=750,
                neighbors="exact", negative_gradient_method="bh",
            )
        points = tsne.fit(data_features)
    
    elif name == "MDS":
        mds = MDS(n_components=int_dim)
        points = mds.fit_transform(data_features)

    elif name == "Isomap":
        isomap = Isomap(n_components=int_dim)
        points = isomap.fit_transform(data_features)


    elif name == "KernelPCA":
        kpca = KernelPCA(n_components=int_dim, kernel='linear')
        points = kpca.fit_transform(data_features)

    elif name == "LLE":
        lle = LocallyLinearEmbedding(n_components=int_dim)
        points = lle.fit_transform(data_features)

    elif name == "HessianLLE":
        lapeig = LocallyLinearEmbedding(n_neighbors = 6, n_components=int_dim, method = 'hessian')
        points = lapeig.fit_transform(data_features)

    elif name == "LaplacianEigenmap":
        lapeig = SpectralEmbedding(n_components=int_dim)
        points = lapeig.fit_transform(data_features)

    elif name == "LargeVis":
        outdim = int_dim
        threads = 24
        samples = -1
        prop = -1
        alpha = -1
        trees = -1
        neg = -1
        neigh = -1
        gamma = -1
        perp = -1

        with open('largevis_input.txt','w') as out:
            out.write("{}\t{}\n".format(*data_features.as_matrix().shape))
            for row in data_features.as_matrix():
                out.write('\t'.join(row.astype(str))+'\n')

        LargeVis.loadfile("largevis_input.txt")
        points = LargeVis.run(outdim, threads, samples, prop, alpha, trees, neg, neigh, gamma, perp)
    
    end = time.time()

    return (end-start)
