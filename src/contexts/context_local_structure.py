
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut

RANDOM_STATE = 42


def calculate_pairwise_distances(df_for_Box_Plot_features, points, distance='euclidean'):
    """
    Computes Pairwise euclidean distances

    Parameters
    ----------
    df_for_Box_Plot_features : list
        original features
    points : nD array
        embedding
    distance: String
        distance, default value is "euclidean"
    Returns
    ----------
    distance_original : nD array
        euclidean distances in the original dataset
    distance_embeddings : nD array
        euclidean distances in the embedding
    """
    distance_original = pdist(df_for_Box_Plot_features, metric=distance)
    distance_embeddings = pdist(points, metric=distance)
    return distance_original, distance_embeddings

def calculate_geodesic_distance(df_for_Box_Plot_features, points):
    """
    Computes Pairwise geodesic distances

    Parameters
    ----------
    df_for_Box_Plot_features : list
        original features
    points : nD array
        embedding

    Returns
    ----------
    geo_distance_original : nD array
        geodesic distances in the original dataset
    geo_distance_embeddings : nD array
        geodesic distances in the embedding
    """
    embedding = Isomap(n_components=2)
    embedding.fit(df_for_Box_Plot_features)
    unsquareform = lambda a: a[np.nonzero(np.triu(a, 1))] ## define a lambda to unsquare the distance matrix
    geo_distance_original = unsquareform(embedding.dist_matrix_) ## get a condensed matrix of pairwise geodesic distance among points    
    
    embedding1 = Isomap(n_components=2)
    embedding1.fit(points)
    embedding1.dist_matrix_[embedding1.dist_matrix_ == 0] = -9999 ## turn all 0 distances to -9999
    geo_distance_embeddings = unsquareform(embedding1.dist_matrix_) ## get a condensed matrix of pairwise geodesic distance among points
    geo_distance_embeddings[geo_distance_embeddings == -9999] = 0 ## turn all -9999 distances back to 0
    
    return geo_distance_original, geo_distance_embeddings

def generate_histograms(distance_original, distance_embeddings, no_of_bins):
    """
    Generates histograms

    Parameters
    ----------
    distance_original : nD array
        original distances
    distance_embeddings : nD array
        embedding distances
    no_of_bins : integer
        number of bins in the histogram
    Returns
    ----------
    bin_edges_original : list
        bin edges
    """
    countsOriginal, bin_edges_original = np.histogram(distance_original, bins = no_of_bins)
    #print("Original Distance Binned Element Counts: ", countsOriginal)
    countsEmbedding, bin_edges_embedding = np.histogram(distance_embeddings, bins = no_of_bins)
    #print("Embedding Distance Binned Element Counts: ", countsEmbedding)
    plt.figure()
    plt.hist(distance_original, bins = no_of_bins)
    plt.show()
    plt.title("Pairwise distances in original data")
    plt.hist(distance_embeddings, bins = no_of_bins)
    plt.show()
    plt.title("Pairwise distances in embeddings")
    return bin_edges_original

def calculate_box_plot_details(distance_original, distance_embeddings, bin_edges_original):
    """
    Computes the details of the Box-plots
    """

    inds_original = np.digitize(distance_original, bins=bin_edges_original)
    ##print("number of bins = ", np.unique(inds_original))
    
    for i in range(1,52):
        globals()["array" + str(i)] = []

    for j in range(0,len(inds_original)):
        globals()["array" + str(inds_original[j])].append(distance_embeddings[j])
        
        
    data_to_plot = [array1, array2, array3, array4, array5, array6, array7, array8, array9, array10,
               array11, array12, array13, array14, array15, array16, array17, array18, array19, array20,
               array21, array22, array23, array24, array25, array26, array27, array28, array29, array30,
               array31, array32, array33, array34, array35, array36, array37, array38, array39, array40,
               array41, array42, array43, array44, array45, array46, array47, array48, array49, array50, array51]
    
    return data_to_plot

def generate_box_plots(data_to_plot):
    """
    Generates Box-plots
    """
    fig = plt.figure(1, figsize=(14, 10))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')

    ## add patch_artist=True option to ax.boxplot() 
    ## to get fill color
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)


def gen_error_1_NN(embedding, labels):
    """
    Computes 1-NN generalization error

    Parameters
    ----------
    embedding : nD array
        embedding
    labels : list
        original labels
    Returns
    ----------
    gen_error : float
       generalization error
    """
    model = KNeighborsClassifier(n_neighbors=1)
    loo = LeaveOneOut()
    loo.get_n_splits(embedding)
    scores = cross_val_score(model , X = embedding , y = labels, cv = loo)
    gen_error = (1 - np.mean(scores))
    
    return gen_error



























