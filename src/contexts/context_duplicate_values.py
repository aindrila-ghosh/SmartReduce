
import numpy as np

def measure_difference_between_embeddingds(algo, original_embedding, duplicate_embedding):
	'''
    computes structural similarity between embeddings with and without duplicate values
    '''

	duplicate_embedding_subset = duplicate_embedding[:200,:]

	difference = np.ceil(np.sum((original_embedding-duplicate_embedding_subset)**2)/1000.)
    
	return difference