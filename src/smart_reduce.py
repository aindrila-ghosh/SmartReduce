import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import pandas as pd
import numpy as np
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import wilcoxon, ttest_rel, friedmanchisquare
from scipy.stats import rankdata
from scipy.stats import spearmanr

## Importing local Python files

import generic_methods, Preprocessing, DR_algorithms
from contexts import accuracy_context, context_time, context_local_structure, context_global_structure, context_duplicate_values, context_outlier_values, context_stability, context_missing_values, context_dimRed
from statistical_tests import mcnemar_test, nonparametric_tests

"""
## "M_Core_tSNE", "openTSNE", "LargeVis", "MVU" are removed from the list of algorithms as they were not added as a part of the package.
## This decision was made to avoid any LICENSE conflicts with the open source packages.
## Users can add any algorithms they wish, but those would need to be added manually.
"""

## Defining global variables

algorithms = ["UMAP", "tSNE", "PCA", "Trimap", "Isomap", "KernelPCA", "LEM", "LTSA", "MDS", "HessianLLE", "LLE", "MLLE"]
metrics = ["K-NN Accuracy", "AUC_lnK_R_NX", "Relative Computational Complexity", "Spearmans Rank Correlation", "Residual Variance", "Q_local", "mean_R_NX", "Q_global", "Global Structure_Triplets", "Global Structure_K_max", "Stability with Missing Values", "Sensitivity to Outliers", "Reproducibility with Partial Records"]
accuracy_tests = ["knn"]
dimRed_tests = ["AUC_lnK_R_NX", "mean_R_NX", "Q_local", "Q_global", "K_max"]
local_distances = ["euclidean"]
global_distances = ["geodesic"]


def get_dataset_names(datasets):
	"""
    Returns the names of the datasets.
    Parameters
    ----------
    datasets : tuple
        tuple of datasets
    Returns
    ----------
    dataset_names : list
        list of dataset names
    """
	dataset_names = []
	for index_dataset, (dataset_name, df_data) in enumerate(datasets):
		dataset_names.append(dataset_name)
	return dataset_names


def compute_algorithm_ranks(datasets):
	"""
    Returns the names of the datasets.
    Parameters
    ----------
    datasets : tuple
        tuple of datasets
    Returns
    ----------
    df_Friedman_Ranking : dataframe
        dataframe of Friedman ranks of each algorithm for each of the computed metrics
    """

	for index_dataset, (dataset_name, df_data) in enumerate(datasets):
        
	    for algo in algorithms:   
	        for test in accuracy_tests:
	            globals()[dataset_name+"_"+algo+"_"+test] = []
	        
	        for test in dimRed_tests:
	            globals()[dataset_name+"_"+algo+"_"+test] = []
	        
	        globals()[dataset_name+"_"+algo+"_time"] = []
	            
	        for measure in local_distances:
	            globals()[dataset_name+"_"+algo+"_"+measure] = []
	        
	        globals()[dataset_name+"_"+algo+"_res_variance"] = []
	            
	        for measure in global_distances:
	            globals()[dataset_name+"_"+algo+"_"+measure] = []
	        
	        globals()[dataset_name+"_"+algo+"_missing"] = []
	        
	        globals()[dataset_name+"_"+algo+"_otlr"] = []
	        
	        globals()[dataset_name+"_"+algo+"_partial_rows"] = []
	  
	    for sample_index in range(0,1): ## This range should vary from 0 to 10,000
	        dataframe_dictionary_outliers = {}
	        cutoff = 2 
	        print("At Iteration: ", sample_index)
	            
	        ## df_sample = df_data.sample(optimum_sample_size[index_dataset]) ## Selecting optimum sample size from dataset
	        df_sub_sample = df_data.sample(200) ## for this demo we make the sample size constant
	        
	        for algo in algorithms:
	            print("Dataset: ", dataset_name)
	            print("Algorithm: ", algo) 
	            
	            if algo == "FIt_SNE":
	                df_sub_sample = df_data           
	            
	            df_sub_sample.reset_index(inplace = True, drop=True)
	            data_features, data_target = generic_methods.set_features_and_target(df_sub_sample)
	            embedding = DR_algorithms.run_DR_Algorithm(algo, data_features, data_target)
	            
	            ## run each accuracy test on the sample
	            accuracy_knn = accuracy_context.train_ML_models(data_target, embedding)
	            globals() [dataset_name+"_"+algo+"_knn"].append(accuracy_knn)
	            print("Accuracy metric calculation finished.....................")

	            ## record execution time
	            exec_time = context_time.run_DR_Algorithm(algo, data_features, data_target)
	            globals() [dataset_name+"_"+algo+"_time"].append(exec_time)
	            print("Execution Time metric calculation finished.....................")
	            
	            ## record dimRed metrics
	            AUC_lnK_R_NX, mean_R_NX, Q_local, Q_global, K_max = context_dimRed.calculate_dimRed_metrics(embedding, df_sub_sample)
	            globals() [dataset_name+"_"+algo+"_AUC_lnK_R_NX"].append(AUC_lnK_R_NX)
	            globals() [dataset_name+"_"+algo+"_mean_R_NX"].append(mean_R_NX)
	            globals() [dataset_name+"_"+algo+"_Q_local"].append(Q_local)
	            globals() [dataset_name+"_"+algo+"_Q_global"].append(Q_global)
	            print("dimRed metrics calculation finished.....................")
	            
	            ## record local distances
	            embedding_corrected = np.nan_to_num(embedding)
	            for measure in local_distances:
	                distance_original, distance_embeddings = context_local_structure.calculate_pairwise_distances(data_features, embedding_corrected)
	                distance_original = np.nan_to_num(distance_original)
	                distance_embeddings = np.nan_to_num(distance_embeddings)
	            rho, p_val = spearmanr(distance_original, distance_embeddings)
	            globals() [dataset_name+"_"+algo+"_"+measure].append(rho)
	            print("Spearman Corr calculation finished.....................")
	            globals() [dataset_name+"_"+algo+"_res_variance"].append(np.corrcoef(distance_original, distance_embeddings)[1,0])
	            print("Residual Variance calculation finished.....................")
	            
	            ## record global distances
	            geodesic_result = context_global_structure.run_n_points_ordering(df_sub_sample, embedding)
	            globals() [dataset_name+"_"+algo+"_geodesic"].append(geodesic_result)
	            print("Global Triplets calculation finished.....................")
	            
	            K_max_agreement = context_global_structure.run_K_max_ordering(data_features, embedding, K_max)
	            globals() [dataset_name+"_"+algo+"_K_max"].append(K_max_agreement)
	            print("K_max agreement calculation finished.....................")
	            
	            ## record stability with missing values
	            
	            nmi = context_missing_values.measure_impact_of_missing_values(df_sub_sample, algo)
	            globals() [dataset_name+"_"+algo+"_missing"].append(nmi)
	            print("NMI calculation finished.....................")
	            
	            ## record sensitivity with outlier values
	            
	            ## check for outliers and add 20% outlier in the data
	            for column in df_data:
	                globals() [column] = []
	                outliers = [0]
	                ## check for outliers using Median Absolute Deviation (MAD)
	                outliers = context_outlier_values.check_for_outlier_values(df_data[column].values, cutoff)
	                outlier_count = 40
	                    
	                for i in range(0,outlier_count):
	                    min_outlier = min(outliers)
	                    max_outlier = max(outliers)
	                    if min_outlier > max_outlier:
	                        min_outlier = max_outlier
	                    globals() [column].append(np.random.random_integers(min_outlier,max_outlier))
	                dataframe_dictionary_outliers.update({column : globals() [column]})

	            df_Outliers = pd.concat([df_sub_sample,pd.DataFrame(dataframe_dictionary_outliers)])
	            outlier_features, outlier_target = generic_methods.set_features_and_target(df_Outliers)
	            try:
	                outlier_embedding = DR_algorithms.run_DR_Algorithm(algo, outlier_features, outlier_target, intrinsic_dimensionality[index_dataset])
	            except:
	                outlier_embedding = embedding_corrected
	            np.nan_to_num(outlier_embedding)
	            
	            difference = context_outlier_values.measure_difference_between_embeddingds(algo, embedding_corrected, outlier_embedding)
	            globals() [dataset_name+"_"+algo+"_otlr"].append(difference)
	            print("SSI calculation finished.....................")
	            
	            ## record reproducibility with partial records
	            differences_partial_rows = []
	            difference_partial_rows = context_stability.run_partial_row_experiments(algo, df_sub_sample, embedding_corrected)
	            globals() [dataset_name+"_"+algo+"_partial_rows"].append(difference_partial_rows)
	            print("Log loss calculation finished.....................")
	            
	            ## end for loop for Algorithms
	        ## end for loop for Samples
	    
	    ## save the results of each accuracy test in individual dataframes
	    print("All iterations completed: Saving the results.....................")
	    for test in accuracy_tests:
	        globals()["dataframe_dictionary_"+test] = {}
	        for algo in algorithms:
	            globals() ["dataframe_dictionary_"+test].update({algo : globals() [dataset_name+"_"+algo+"_"+test]})
	   
	    globals() ["df_knn_Accuracy_"+dataset_name] = pd.DataFrame(dataframe_dictionary_knn)
	      
	    for test in accuracy_tests:
	        globals() ["mean_"+test+"_"+dataset_name] = {}
	        globals() ["std_"+test+"_"+dataset_name] = {}
	    
	    for test in accuracy_tests:
	        for column in globals()["df_knn_Accuracy_"+dataset_name]:
	            globals() ["mean_"+test+"_"+dataset_name].update({column : np.mean(globals()["df_knn_Accuracy_"+dataset_name][column])})
	            globals() ["std_"+test+"_"+dataset_name].update({column : np.std(globals()["df_knn_Accuracy_"+dataset_name][column])})
	    
	            
	    ## save the results of execution time in individual dataframes
	    dataframe_dictionary_time = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_time.update({algo : globals() [dataset_name+"_"+algo+"_time"]})
	            
	    globals() ["df_Time_"+dataset_name] = pd.DataFrame(dataframe_dictionary_time)
	    
	    globals() ["mean_time_"+dataset_name] = {}
	    globals() ["std_time_"+dataset_name] = {} 
	    
	    for column in globals()["df_Time_"+dataset_name]:
	        globals() ["mean_time_"+dataset_name].update({column : np.mean(globals()["df_Time_"+dataset_name][column])})
	        globals() ["std_time_"+dataset_name].update({column : np.std(globals()["df_Time_"+dataset_name][column])})
	    
	    
	    ## save the results of each dimRed metric test in individual dataframes
	    for test in dimRed_tests:
	        if test != 'K_max':
	            globals()["dataframe_dictionary_"+test] = {}
	            for algo in algorithms:
	                globals() ["dataframe_dictionary_"+test].update({algo : globals() [dataset_name+"_"+algo+"_"+test]})
	   
	    globals() ["df_AUC_lnK_R_NX_"+dataset_name] = pd.DataFrame(dataframe_dictionary_AUC_lnK_R_NX)
	    globals() ["df_mean_R_NX_"+dataset_name] = pd.DataFrame(dataframe_dictionary_mean_R_NX)
	    globals() ["df_Q_local_"+dataset_name] = pd.DataFrame(dataframe_dictionary_Q_local)
	    globals() ["df_Q_global_"+dataset_name] = pd.DataFrame(dataframe_dictionary_Q_global)
	    
	        
	    for test in dimRed_tests:
	        if test != 'K_max':
	            globals() ["mean_"+test+"_"+dataset_name] = {}
	            globals() ["std_"+test+"_"+dataset_name] = {}
	    
	    for test in dimRed_tests:
	        if test != 'K_max':
	            for column in globals()["df_"+test+"_"+dataset_name]:
	                globals() ["mean_"+test+"_"+dataset_name].update({column : np.mean(globals()["df_"+test+"_"+dataset_name][column])})
	                globals() ["std_"+test+"_"+dataset_name].update({column : np.std(globals()["df_"+test+"_"+dataset_name][column])})

	    
	    ## save the results of preservation of local distances in individual dataframes
	    for measure in local_distances:
	        globals()["dataframe_dictionary_"+measure+"_"+dataset_name] = {}
	        for algo in algorithms:
	            globals() ["dataframe_dictionary_"+measure+"_"+dataset_name].update({algo : globals() [dataset_name+"_"+algo+"_"+measure]})
	        globals() ["df_pairwise_"+measure+"_distances"+"_"+dataset_name]  =  pd.DataFrame(globals() ["dataframe_dictionary_"+measure+"_"+dataset_name])
	    
	    for measure in local_distances:
	        globals() ["mean_"+measure+"_"+dataset_name] = {}
	        globals() ["std_"+measure+"_"+dataset_name] = {}
	        
	    for measure in local_distances:
	        for column in globals()["df_pairwise_"+measure+"_distances"+"_"+dataset_name]:
	            globals() ["mean_"+measure+"_"+dataset_name].update({column : np.mean(globals()["df_pairwise_"+measure+"_distances"+"_"+dataset_name][column])})
	            globals() ["std_"+measure+"_"+dataset_name].update({column : np.std(globals()["df_pairwise_"+measure+"_distances"+"_"+dataset_name][column])})

	    
	    ## save the results of residual variance in individual dataframes
	    dataframe_dictionary_res_variance = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_res_variance.update({algo : globals() [dataset_name+"_"+algo+"_res_variance"]})
	            
	    globals() ["df_res_var_"+dataset_name] = pd.DataFrame(dataframe_dictionary_res_variance)
	    
	    globals() ["mean_res_var_"+dataset_name] = {}
	    globals() ["std_res_var_"+dataset_name] = {} 
	    
	    for column in globals()["df_res_var_"+dataset_name]:
	        globals() ["mean_res_var_"+dataset_name].update({column : np.mean(globals()["df_res_var_"+dataset_name][column])})
	        globals() ["std_res_var_"+dataset_name].update({column : np.std(globals()["df_res_var_"+dataset_name][column])})  
	    
	    
	    ## save the results of preservation of global distances in individual dataframes
	    dataframe_dictionary_geodesic = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_geodesic.update({algo : globals() [dataset_name+"_"+algo+"_geodesic"]})
	    globals() ["df_n_points_ordering_geodesic_"+dataset_name] = pd.DataFrame(dataframe_dictionary_geodesic)
	      
	    globals() ["mean_n_points_ordering_geodesic_"+dataset_name] = {}
	    globals() ["std_n_points_ordering_geodesic_"+dataset_name] = {} 
	    
	    for column in globals()["df_n_points_ordering_geodesic_"+dataset_name]:
	        globals() ["mean_n_points_ordering_geodesic_"+dataset_name].update({column : np.mean(globals()["df_n_points_ordering_geodesic_"+dataset_name][column])})
	        globals() ["std_n_points_ordering_geodesic_"+dataset_name].update({column : np.std(globals()["df_n_points_ordering_geodesic_"+dataset_name][column])})
	    
	    dataframe_dictionary_K_max = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_K_max.update({algo : globals() [dataset_name+"_"+algo+"_K_max"]})
	    globals() ["df_K_max_agreement_"+dataset_name] = pd.DataFrame(dataframe_dictionary_K_max)
	      
	    globals() ["mean_K_max_agreement_"+dataset_name] = {}
	    globals() ["std_K_max_agreement_"+dataset_name] = {} 
	    
	    for column in globals()["df_K_max_agreement_"+dataset_name]:
	        globals() ["mean_K_max_agreement_"+dataset_name].update({column : np.mean(globals()["df_K_max_agreement_"+dataset_name][column])})
	        globals() ["std_K_max_agreement_"+dataset_name].update({column : np.std(globals()["df_K_max_agreement_"+dataset_name][column])})
	    
	    
	    ## save the results of stability with duplicate values in individual dataframes
	    dataframe_dictionary_difference = {}
	    
	    for algo in algorithms:
	         dataframe_dictionary_difference.update({algo : globals() [dataset_name+"_"+algo+"_missing"]})
	            
	    globals() ["df_Difference_Duplicate"+dataset_name] = pd.DataFrame(dataframe_dictionary_difference)
	    
	    globals() ["mean_difference_duplicate"+dataset_name] = {}
	    globals() ["std_difference_duplicate"+dataset_name] = {} 
	    
	    for column in globals()["df_Difference_Duplicate"+dataset_name]:
	        globals() ["mean_difference_duplicate"+dataset_name].update({column : np.mean(globals()["df_Difference_Duplicate"+dataset_name][column])})
	        globals() ["std_difference_duplicate"+dataset_name].update({column : np.std(globals()["df_Difference_Duplicate"+dataset_name][column])})

	     ## save the results of sensitivity with outlier values in individual dataframes
	    dataframe_dictionary_difference = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_difference.update({algo : globals() [dataset_name+"_"+algo+"_otlr"]})
	            
	    globals() ["df_Difference_Outlier"+dataset_name] = pd.DataFrame(dataframe_dictionary_difference)
	    
	    globals() ["mean_difference_outlier"+dataset_name] = {}
	    globals() ["std_difference_outlier"+dataset_name] = {} 
	    
	    for column in globals()["df_Difference_Outlier"+dataset_name]:
	        globals() ["mean_difference_outlier"+dataset_name].update({column : np.mean(globals()["df_Difference_Outlier"+dataset_name][column])})
	        globals() ["std_difference_outlier"+dataset_name].update({column : np.std(globals()["df_Difference_Outlier"+dataset_name][column])})
	    

	    ## save the results of reproducibility with partial records in individual dataframes
	    dataframe_dictionary_partial_rows = {}
	    
	    for algo in algorithms:
	        dataframe_dictionary_partial_rows.update({algo : globals() [dataset_name+"_"+algo+"_partial_rows"]})
	        
	    globals() ["df_Difference_Partial_Rows_"+dataset_name] = pd.DataFrame(dataframe_dictionary_partial_rows)
	    
	    globals() ["mean_Partial_Rows_"+dataset_name] = {}
	    globals() ["std_Partial_Rows_"+dataset_name] = {} 
	    
	    for column in globals()["df_Difference_Partial_Rows_"+dataset_name]:
	        globals() ["mean_Partial_Rows_"+dataset_name].update({column : np.mean(globals()["df_Difference_Partial_Rows_"+dataset_name][column])})
	        globals() ["std_Partial_Rows_"+dataset_name].update({column : np.std(globals()["df_Difference_Partial_Rows_"+dataset_name][column])})

	df_Friedman_Ranking = create_overall_dataframes(datasets)
	return df_Friedman_Ranking

def create_overall_dataframes(datasets):
	"""
    Creates overall dataframes holding the performance scores of each algorithm in each metric.
    Parameters
    ----------
    datasets : tuple
        tuple of datasets
    Returns
    ----------
    df_Friedman_Ranking : dataframe
        dataframe of Friedman ranks of each algorithm for each of the computed metrics
    """

	dataset_names = get_dataset_names(datasets)
	## Creating overall dataframes

	overall_accuracy = []
	overall_AUC_lnK_R_NX = []
	overall_time = []
	overall_distance = []
	overall_res_var = []
	overall_mean_R_NX = []
	overall_Q_local = []
	overall_n_points_ordering = []
	overall_Q_global = []
	overall_K_max_agreement = []
	overall_difference_missing = []
	overall_difference_outlier = []
	overall_distance_rows = []


	## creating dictionaries for holding mean

	all_means_accuracy = {}
	all_means_AUC_lnK_R_NX = {}
	all_means_time = {}
	all_means_local_struc = {}
	all_means_res_var = {}
	all_means_mean_R_NX = {}
	all_means_Q_local = {}
	all_means_global_struc = {}
	all_means_Q_global = {}
	all_means_K_max_agreement = {}
	all_means_missing = {}
	all_means_outliers = {}
	all_means_partial_rec = {}


	## Add other steps

	for algo in algorithms:
	    globals() [algo+"_accuracy"] = []
	    globals() [algo+"_AUC_lnK_R_NX"] = []
	    globals() [algo+"_time"] = []
	    globals() [algo+"_local_strc"] = []
	    globals() [algo+"_res_var"] = []
	    globals() [algo+"_mean_R_NX"] = []
	    globals() [algo+"_Q_local"] = []
	    globals() [algo+"_global_strc"] = []
	    globals() [algo+"_Q_global"] = []
	    globals() [algo+"_K_max_agreement"] = []
	    globals() [algo+"_missing"] = []
	    globals() [algo+"_outliers"] = []
	    globals() [algo+"_partial_rec"] = []

	    
	    for index_dataset, (dataset_name, df_data) in enumerate(datasets):
	        globals() [algo+"_accuracy"].append(globals() ["mean_knn_"+dataset_name][algo])
	        globals() [algo+"_AUC_lnK_R_NX"].append(globals() ["mean_AUC_lnK_R_NX_"+dataset_name][algo])
	        globals() [algo+"_time"].append(globals() ["mean_time_"+dataset_name][algo])
	        globals() [algo+"_local_strc"].append(globals() ["mean_euclidean_"+dataset_name][algo])
	        globals() [algo+"_res_var"].append(globals() ["mean_res_var_"+dataset_name][algo])
	        globals() [algo+"_mean_R_NX"].append(globals() ["mean_mean_R_NX_"+dataset_name][algo])
	        globals() [algo+"_Q_local"].append(globals() ["mean_Q_local_"+dataset_name][algo])
	        globals() [algo+"_global_strc"].append(globals() ["mean_n_points_ordering_geodesic_"+dataset_name][algo])
	        globals() [algo+"_Q_global"].append(globals() ["mean_Q_global_"+dataset_name][algo])
	        globals() [algo+"_K_max_agreement"].append(globals() ["mean_K_max_agreement_"+dataset_name][algo])
	        globals() [algo+"_missing"].append(globals() ["mean_difference_duplicate"+dataset_name][algo])
	        globals() [algo+"_outliers"].append(globals() ["mean_difference_outlier"+dataset_name][algo])
	        globals() [algo+"_partial_rec"].append(globals() ["mean_Partial_Rows_"+dataset_name][algo])

	        
	    all_means_accuracy.update({algo: np.mean(globals() [algo+"_accuracy"])})
	    all_means_AUC_lnK_R_NX.update({algo: np.mean(globals() [algo+"_AUC_lnK_R_NX"])})
	    all_means_time.update({algo: np.mean(globals() [algo+"_time"])})
	    all_means_local_struc.update({algo: np.mean(globals() [algo+"_local_strc"])})
	    all_means_res_var.update({algo: np.mean(globals() [algo+"_res_var"])})
	    all_means_mean_R_NX.update({algo: np.mean(globals() [algo+"_mean_R_NX"])})
	    all_means_Q_local.update({algo: np.mean(globals() [algo+"_Q_local"])})
	    all_means_global_struc.update({algo: np.mean(globals() [algo+"_global_strc"])})
	    all_means_Q_global.update({algo: np.mean(globals() [algo+"_Q_global"])})
	    all_means_K_max_agreement.update({algo: np.mean(globals() [algo+"_K_max_agreement"])})
	    all_means_missing.update({algo: np.mean(globals() [algo+"_missing"])})
	    all_means_outliers.update({algo: np.mean(globals() [algo+"_outliers"])})
	    all_means_partial_rec.update({algo: np.mean(globals() [algo+"_partial_rec"])})


	    overall_accuracy.append(globals() [algo+"_accuracy"])
	    overall_AUC_lnK_R_NX.append(globals() [algo+"_AUC_lnK_R_NX"])
	    overall_time.append(globals() [algo+"_time"])
	    overall_distance.append(globals() [algo+"_local_strc"])
	    overall_res_var.append(globals() [algo+"_res_var"])
	    overall_mean_R_NX.append(globals() [algo+"_mean_R_NX"])
	    overall_Q_local.append(globals() [algo+"_Q_local"])
	    overall_n_points_ordering.append(globals() [algo+"_global_strc"])
	    overall_Q_global.append(globals() [algo+"_Q_global"])
	    overall_K_max_agreement.append(globals() [algo+"_K_max_agreement"])
	    overall_difference_missing.append(globals() [algo+"_missing"])
	    overall_difference_outlier.append(globals() [algo+"_outliers"])
	    overall_distance_rows.append(globals() [algo+"_partial_rec"])


	df_overall_accuracy = pd.DataFrame(overall_accuracy)
	df_overall_accuracy.columns = dataset_names 
	df_overall_accuracy["Algorithm"] = algorithms
	df_overall_accuracy = df_overall_accuracy.set_index("Algorithm")

	df_overall_AUC_lnK_R_NX = pd.DataFrame(overall_AUC_lnK_R_NX)
	df_overall_AUC_lnK_R_NX.columns = dataset_names 
	df_overall_AUC_lnK_R_NX["Algorithm"] = algorithms
	df_overall_AUC_lnK_R_NX = df_overall_AUC_lnK_R_NX.set_index("Algorithm")

	df_overall_time = pd.DataFrame(overall_time)
	df_overall_time.columns = dataset_names 
	df_overall_time["Algorithm"] = algorithms
	df_overall_time = df_overall_time.set_index("Algorithm")

	df_overall_distance = pd.DataFrame(overall_distance)
	df_overall_distance.columns = dataset_names
	df_overall_distance["Algorithm"] = algorithms
	df_overall_distance = df_overall_distance.set_index("Algorithm")

	df_overall_res_var = pd.DataFrame(overall_res_var)
	df_overall_res_var.columns = dataset_names
	df_overall_res_var["Algorithm"] = algorithms
	df_overall_res_var = df_overall_res_var.set_index("Algorithm")

	df_overall_mean_R_NX = pd.DataFrame(overall_mean_R_NX)
	df_overall_mean_R_NX.columns = dataset_names
	df_overall_mean_R_NX["Algorithm"] = algorithms
	df_overall_mean_R_NX = df_overall_mean_R_NX.set_index("Algorithm")

	df_overall_Q_local = pd.DataFrame(overall_Q_local)
	df_overall_Q_local.columns = dataset_names
	df_overall_Q_local["Algorithm"] = algorithms
	df_overall_Q_local = df_overall_Q_local.set_index("Algorithm")

	df_overall_n_points_ordering = pd.DataFrame(overall_n_points_ordering)
	df_overall_n_points_ordering.columns = dataset_names
	df_overall_n_points_ordering["Algorithm"] = algorithms
	df_overall_n_points_ordering = df_overall_n_points_ordering.set_index("Algorithm")

	df_overall_Q_global = pd.DataFrame(overall_Q_global)
	df_overall_Q_global.columns = dataset_names
	df_overall_Q_global["Algorithm"] = algorithms
	df_overall_Q_global = df_overall_Q_global.set_index("Algorithm")

	df_overall_K_max_agreement = pd.DataFrame(overall_K_max_agreement)
	df_overall_K_max_agreement.columns = dataset_names
	df_overall_K_max_agreement["Algorithm"] = algorithms
	df_overall_K_max_agreement = df_overall_K_max_agreement.set_index("Algorithm")

	df_overall_difference_missing = pd.DataFrame(overall_difference_missing)
	df_overall_difference_missing.columns = dataset_names 
	df_overall_difference_missing["Algorithm"] = algorithms
	df_overall_difference_missing = df_overall_difference_missing.set_index("Algorithm")

	df_overall_difference_outlier = pd.DataFrame(overall_difference_outlier)
	df_overall_difference_outlier.columns = dataset_names
	df_overall_difference_outlier["Algorithm"] = algorithms
	df_overall_difference_outlier = df_overall_difference_outlier.set_index("Algorithm")

	df_overall_distance_rows = pd.DataFrame(overall_distance_rows)
	df_overall_distance_rows.columns = dataset_names ## Need to create a list with these or type them
	df_overall_distance_rows["Algorithm"] = algorithms
	df_overall_distance_rows = df_overall_distance_rows.set_index("Algorithm")

	df_Friedman_Ranking = order_algorithms(all_means_accuracy, all_means_AUC_lnK_R_NX, all_means_time, all_means_local_struc, all_means_res_var, all_means_mean_R_NX, all_means_Q_local, all_means_global_struc, all_means_Q_global, all_means_K_max_agreement, all_means_missing, all_means_outliers, all_means_partial_rec, df_overall_accuracy, df_overall_AUC_lnK_R_NX, df_overall_time, df_overall_distance, df_overall_res_var, df_overall_mean_R_NX, df_overall_Q_local, df_overall_n_points_ordering, df_overall_Q_global, df_overall_K_max_agreement, df_overall_difference_missing, df_overall_difference_outlier, df_overall_distance_rows)
	return df_Friedman_Ranking

def order_algorithms(all_means_accuracy, all_means_AUC_lnK_R_NX, all_means_time, all_means_local_struc, all_means_res_var, all_means_mean_R_NX, all_means_Q_local, all_means_global_struc, all_means_Q_global, all_means_K_max_agreement, all_means_missing, all_means_outliers, all_means_partial_rec, df_overall_accuracy, df_overall_AUC_lnK_R_NX, df_overall_time, df_overall_distance, df_overall_res_var, df_overall_mean_R_NX, df_overall_Q_local, df_overall_n_points_ordering, df_overall_Q_global, df_overall_K_max_agreement, df_overall_difference_missing, df_overall_difference_outlier, df_overall_distance_rows):
	"""
    Orders the algorithms based on their performances in each metric.
    Parameters
    ----------
    all_means_accuracy, all_means_AUC_lnK_R_NX, all_means_time, all_means_local_struc, all_means_res_var, all_means_mean_R_NX, all_means_Q_local, all_means_global_struc, all_means_Q_global, all_means_K_max_agreement, all_means_missing, all_means_outliers, all_means_partial_rec) : dictionaries
        mean performance scores of all algorithms on all datasets
    df_overall_accuracy, df_overall_AUC_lnK_R_NX, df_overall_time, df_overall_distance, df_overall_res_var, df_overall_mean_R_NX, df_overall_Q_local, df_overall_n_points_ordering, df_overall_Q_global, df_overall_K_max_agreement, df_overall_difference_missing, df_overall_difference_outlier, df_overall_distance_rows: dataframes
    	DataFrames of performance scores
    Returns
    ----------
    df_Friedman_Ranking : dataframe
        dataframe of Friedman ranks of each algorithm for each of the computed metrics
    """

	Accuracy_ranks = sorted(all_means_accuracy.items(), key=lambda x: x[1])
	AUC_lnK_R_NX_ranks = sorted(all_means_AUC_lnK_R_NX.items(), key=lambda x: x[1])
	Time_ranks = sorted(all_means_time.items(), key=lambda x: x[1], reverse=True)
	Local_structure_ranks = sorted(all_means_local_struc.items(), key=lambda x: x[1])
	Residual_Variance_ranks = sorted(all_means_res_var.items(), key=lambda x: x[1])
	mean_R_NX_ranks = sorted(all_means_mean_R_NX.items(), key=lambda x: x[1])
	Q_local_ranks = sorted(all_means_Q_local.items(), key=lambda x: x[1])
	Global_structure_ranks = sorted(all_means_global_struc.items(), key=lambda x: x[1])
	Q_global_ranks = sorted(all_means_Q_global.items(), key=lambda x: x[1])
	K_max_agreement_ranks = sorted(all_means_K_max_agreement.items(), key=lambda x: x[1])
	Missing_ranks = sorted(all_means_missing.items(), key=lambda x: x[1])
	Outlier_ranks = sorted(all_means_outliers.items(), key=lambda x: x[1])
	Partial_record_Ranks = sorted(all_means_partial_rec.items(), key=lambda x: x[1])

	print("Sorted based on mean accuracy")
	for index in range(0, len(Accuracy_ranks)):
	    print(Accuracy_ranks[index])
	print(".............................")
	print("Sorted based on mean AUC_lnK_R_NX")
	for index in range(0, len(AUC_lnK_R_NX_ranks)):
	    print(AUC_lnK_R_NX_ranks[index])
	print(".............................")
	print("Sorted based on mean time")
	for index in range(0, len(Time_ranks)):
	    print(Time_ranks[index])
	print(".............................")
	print("Sorted based on mean score on preservation of local structure")
	for index in range(0, len(Local_structure_ranks)):
	    print(Local_structure_ranks[index])
	print(".............................")
	print("Sorted based on mean score on Residual Variance")
	for index in range(0, len(Residual_Variance_ranks)):
	    print(Residual_Variance_ranks[index])
	print(".............................")
	print("Sorted based on mean_R_NX")
	for index in range(0, len(mean_R_NX_ranks)):
	    print(mean_R_NX_ranks[index])
	print(".............................")
	print("Sorted based on mean Q_local")
	for index in range(0, len(Q_local_ranks)):
	    print(Q_local_ranks[index])
	print(".............................")
	print("Sorted based on mean score on preservation of global structure")
	for index in range(0, len(Global_structure_ranks)):
	    print(Global_structure_ranks[index])
	print(".............................")
	print("Sorted based on mean Q_global")
	for index in range(0, len(Q_global_ranks)):
	    print(Q_global_ranks[index])
	print(".............................")
	print("Sorted based on mean K_max_agreement")
	for index in range(0, len(K_max_agreement_ranks)):
	    print(K_max_agreement_ranks[index])
	print(".............................")
	print("Sorted based on mean stability with missing values")
	for index in range(0, len(Missing_ranks)):
	    print(Missing_ranks[index])
	print(".............................")
	print("Sorted based on mean sensitivity to outlier scores")
	for index in range(0, len(Outlier_ranks)):
	    print(Outlier_ranks[index])
	print(".............................")
	print("Sorted based on mean reproducibility with partial records score")
	for index in range(0, len(Partial_record_Ranks)):
	    print(Partial_record_Ranks[index])
	print(".............................")

	df_Friedman_Ranking = compute_ranks(df_overall_accuracy, df_overall_AUC_lnK_R_NX, df_overall_time, df_overall_distance, df_overall_res_var, df_overall_mean_R_NX, df_overall_Q_local, df_overall_n_points_ordering, df_overall_Q_global, df_overall_K_max_agreement, df_overall_difference_missing, df_overall_difference_outlier, df_overall_distance_rows)
	return df_Friedman_Ranking

def compute_ranks(df_overall_accuracy, df_overall_AUC_lnK_R_NX, df_overall_time, df_overall_distance, df_overall_res_var, df_overall_mean_R_NX, df_overall_Q_local, df_overall_n_points_ordering, df_overall_Q_global, df_overall_K_max_agreement, df_overall_difference_missing, df_overall_difference_outlier, df_overall_distance_rows):

	"""
    Returns the Friedman ranks of the algorithms for all the datasets.
    
    Returns
    ----------
    df_Friedman_Ranking : dataframe
        dataframe of Friedman ranks of each algorithm for each of the computed metrics
    """

	temp_ranks = np.zeros((len(df_overall_accuracy.T), len(df_overall_accuracy)))

	acc_ranks = []
	for index in range (0,len(df_overall_accuracy.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_accuracy.T.as_matrix()[index]])
	for index in range (0,len(df_overall_accuracy)):
	    acc_ranks.append(np.mean(temp_ranks[:,index]))
	    
	auc_ranks = []
	for index in range (0,len(df_overall_AUC_lnK_R_NX.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_AUC_lnK_R_NX.T.as_matrix()[index]])
	for index in range (0,len(df_overall_AUC_lnK_R_NX)):
	    auc_ranks.append(np.mean(temp_ranks[:,index]))
	    
	time_ranks = []
	for index in range (0,len(df_overall_time.T)):
	    temp_ranks[index,:] = rankdata(df_overall_time.T.as_matrix()[index])
	for index in range (0,len(df_overall_time)):
	    time_ranks.append(np.mean(temp_ranks[:,index]))
	    
	local_ranks = []
	for index in range (0,len(df_overall_distance.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_distance.T.as_matrix()[index]])
	for index in range (0,len(df_overall_distance)):
	    local_ranks.append(np.mean(temp_ranks[:,index]))
	    
	res_var_ranks = []
	for index in range (0,len(df_overall_res_var.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_res_var.T.as_matrix()[index]])
	for index in range (0,len(df_overall_res_var)):
	    res_var_ranks.append(np.mean(temp_ranks[:,index]))

	rnx_ranks = []
	for index in range (0,len(df_overall_mean_R_NX.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_mean_R_NX.T.as_matrix()[index]])
	for index in range (0,len(df_overall_mean_R_NX)):
	    rnx_ranks.append(np.mean(temp_ranks[:,index]))
	    
	Qlocal_ranks = []
	for index in range (0,len(df_overall_Q_local.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_Q_local.T.as_matrix()[index]])
	for index in range (0,len(df_overall_Q_local)):
	    Qlocal_ranks.append(np.mean(temp_ranks[:,index]))
	    
	global_ranks = []
	for index in range (0,len(df_overall_n_points_ordering.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_n_points_ordering.T.as_matrix()[index]])
	for index in range (0,len(df_overall_n_points_ordering)):
	    global_ranks.append(np.mean(temp_ranks[:,index]))
	    
	Qglobal_ranks = []
	for index in range (0,len(df_overall_Q_global.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_Q_global.T.as_matrix()[index]])
	for index in range (0,len(df_overall_Q_global)):
	    Qglobal_ranks.append(np.mean(temp_ranks[:,index]))
	    
	Kmax_ranks = []
	for index in range (0,len(df_overall_K_max_agreement.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_K_max_agreement.T.as_matrix()[index]])
	for index in range (0,len(df_overall_K_max_agreement)):
	    Kmax_ranks.append(np.mean(temp_ranks[:,index]))
	    
	missing_ranks = []
	for index in range (0,len(df_overall_difference_missing.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_difference_missing.T.as_matrix()[index]])
	for index in range (0,len(df_overall_difference_missing)):
	    missing_ranks.append(np.mean(temp_ranks[:,index]))
	    
	outlier_ranks = []
	for index in range (0,len(df_overall_difference_outlier.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_difference_outlier.T.as_matrix()[index]])
	for index in range (0,len(df_overall_difference_outlier)):
	    outlier_ranks.append(np.mean(temp_ranks[:,index]))
	    
	partial_ranks = []
	for index in range (0,len(df_overall_distance_rows.T)):
	    temp_ranks[index,:] = rankdata([-1 * i for i in df_overall_distance_rows.T.as_matrix()[index]])
	for index in range (0,len(df_overall_distance_rows)):
	    partial_ranks.append(np.mean(temp_ranks[:,index]))
	    
	## Creating a dataframe to display the results    
	ranking_data = np.array([acc_ranks, auc_ranks, time_ranks, local_ranks, res_var_ranks, rnx_ranks, Qlocal_ranks, global_ranks, Qglobal_ranks, Kmax_ranks, missing_ranks, outlier_ranks, partial_ranks])
	df_Friedman_Ranking = pd.DataFrame(data=ranking_data,
	                                   index = metrics,
	                                    columns=algorithms)

	return df_Friedman_Ranking


def run_statistical_tests(algorithm_1, algorithm_2, metric_dataframe):

	"""
    Runs Parametric and non-parametric statistical tests.
    Parameters
    ----------
    algorithm_1 : String
        name of an algorithm
    algorithm_2 : String
    	name of another algorithm
    metric_dataframe : DataFrame
    	the overall dataframe for the metric to compare
    Returns
    ----------
    test-scores, p-values, Rp and Re scores of the algorithms
    """
    
    #### Declare lists for plotting and identify best algorithms for pairwise statsitical tests
    
	overall_average_p_values_for_each_bias_wilcoxon = []
	overall_number_of_test_failure_for_each_bias_wilcoxon = []
	overall_Rp_for_each_bias_wilcoxon = []
	overall_Re_for_each_bias_wilcoxon = []

	overall_average_p_values_for_each_bias_paired_t_test = []
	overall_number_of_test_failure_for_each_bias_paired_t_test = []
	overall_Rp_for_each_bias_paired_t_test = []
	overall_Re_for_each_bias_paired_t_test = []

	overall_average_p_values_for_each_bias_exact_McNemar = []
	overall_number_of_test_failure_for_each_bias_exact_McNemar = []
	overall_Rp_for_each_bias_exact_McNemar = []
	overall_Re_for_each_bias_exact_McNemar = []

	overall_average_p_values_for_each_bias_asym_McNemar = []
	overall_number_of_test_failure_for_each_bias_asym_McNemar = []
	overall_Rp_for_each_bias_asym_McNemar = []
	overall_Re_for_each_bias_asym_McNemar = []

	overall_average_p_values_for_each_bias_Friedman = []
	overall_number_of_test_failure_for_each_bias_Friedman = []
	overall_Rp_for_each_bias_Friedman = []
	overall_Re_for_each_bias_Friedman = []

	overall_number_of_test_failure_for_each_bias_Nemenyi = []
	overall_number_of_test_failure_for_each_bias_Holm = []
	overall_number_of_test_failure_for_each_bias_Shaffer = []
	overall_number_of_test_failure_for_each_bias_Unadjusted = []

	all_dataset_names = []
	alpha = 0.05

	for k in range (0,21): ## the bias ranges from bias from 0 to 21

	    print("k=", k)
	    probabilities = []

	    for index_dataset, (dataset_name, df_data) in enumerate(datasets):
	        all_dataset_names.append(dataset_name)

	            ## calculate dataset selection probability

	        try:
	            prb = 1 / ( 1 + math.exp (- 
	                            ( k * (df_overall_accuracy.loc[ algorithm_1 , dataset_name ] 
	                             - df_overall_accuracy.loc[ algorithm_2 , dataset_name ] ) ) ) ) 
	        except OverflowError:
	            prb = 1

	        probabilities.append(prb)

	        ## declaring temporary lists for each test

	    p_values_wilcoxon = []
	    test_result_wilcoxon = []
	    p_values_ttest = []
	    test_result_ttest = []
	    p_value_mcnemar_std = []
	    test_result_mcnemar_std = []
	    p_value_mcnemar_midp = []
	    test_result_mcnemar_midp = []
	    p_value_Friedman = []
	    test_result_Friedman = []
	    test_result_nemenyi = []
	    test_result_holm = []
	    test_result_shaffer = []
	    test_result_unadjusted = []

	    for i in range (0,10): ## change this range from 0 to 100 (as we will select 10 datasets at random, 100 times)

	        ## select 3 datasets at random with given probabilities
	        chosen_datasets = np.random.choice(all_dataset_names, 3, probabilities)

	        for dataset in chosen_datasets:

	            dict_pivot = {}

	                ## run wilcoxon's signed rank test

	            stat, p = wilcoxon(globals() [metric_dataframe+dataset][algorithm_1], globals() [metric_dataframe+dataset][algorithm_2])
	            p_values_wilcoxon.append(p)
	            if p < alpha:
	                test_result_wilcoxon.append("failed")
	            else:
	                test_result_wilcoxon.append("pass")

	            ## run paired t-test

	            stat, p = ttest_rel(globals() [metric_dataframe+dataset][algorithm_1], globals() [metric_dataframe+dataset][algorithm_2])
	            p_values_ttest.append(p)
	            if p < alpha:
	                test_result_ttest.append("failed")
	            else:
	                test_result_ttest.append("pass")

	            ## run McNemar's test

	            win_algo1 = 0
	            win_algo2 = 0

	            for index in range (0,len(globals() [metric_dataframe+dataset][algorithm_1])):
	            	if globals() [metric_dataframe+dataset][algorithm_1][index] >= globals() [metric_dataframe+dataset][algorithm_2][index]:
	                	win_algo1 = win_algo1 + 1
	            	else:
	                	win_algo2 = win_algo2 + 1

	            ## standard McNemar
	            p = mcnemar_test.mcnemar_p(win_algo1, win_algo2)
	            p_value_mcnemar_std.append(p) 
	            if p < alpha:
	                test_result_mcnemar_std.append("failed")
	            else:
	                test_result_mcnemar_std.append("pass")

	                ## Run overall tests - Friedman's test

	            stat, p = friedmanchisquare(*[grp for idx, grp in globals() [metric_dataframe+dataset].iteritems()])
	            p_value_Friedman.append(p)
	            if p < alpha:
	                test_result_Friedman.append("failed")
	            else:
	                test_result_Friedman.append("pass")

	                ## Now Run post hoc tests for each dataset

	                ## calcualate pivots for each column
	            chi2, p, ranking, pivot = nonparametric_tests.friedman_aligned_ranks_test(*[grp for idx, grp in globals() [metric_dataframe+dataset].iteritems()])

	                ## create a dictionary of pivots

	            for ind, column in enumerate(globals() [metric_dataframe+dataset].columns):
	                dict_pivot.update({column : pivot[ind]})

	                ## run Nemenyi Correction Test
	            comparions, z, p, adj_p = nonparametric_tests.nemenyi_multitest(dict_pivot)

	            for p_val in adj_p:
	                if p_val < alpha:
	                    test_result_nemenyi.append("failed")
	                else:
	                    test_result_nemenyi.append("pass")

	                ## run Holm Correction Test
	            comparions, z, p, adj_p = nonparametric_tests.holm_multitest(dict_pivot)

	            for p_val in adj_p:
	                if p_val < alpha:
	                    test_result_holm.append("failed")
	                else:
	                    test_result_holm.append("pass")

	                ## run Shaffer Correction Test
	            comparions, z, p, adj_p = nonparametric_tests.shaffer_multitest(dict_pivot)

	            for p_val in adj_p:
	                if p_val < alpha:
	                    test_result_shaffer.append("failed")
	                else:
	                    test_result_shaffer.append("pass")


	                ## Overall number of failures Unadjusted

	            for p_val in p:
	                if p_val < alpha:
	                    test_result_unadjusted.append("failed")
	                else:
	                    test_result_unadjusted.append("pass")

	        ## for each bias value (i.e., for each k) save the average p values for each test, number of test failures for each test
	        ## also, save the R(p) and R(e) replicability values

	        ## Wilcoxon's signed rank test

	    overall_average_p_values_for_each_bias_wilcoxon.append(np.average(p_values_wilcoxon))
	    overall_number_of_test_failure_for_each_bias_wilcoxon.append(test_result_wilcoxon.count("failed"))

	    Re = ((test_result_wilcoxon.count("pass") * (test_result_wilcoxon.count("pass") - 1)) 
	                    + (test_result_wilcoxon.count("failed") * (test_result_wilcoxon.count("failed") - 1))) / (len(test_result_wilcoxon) * (len(test_result_wilcoxon) - 1))
	    Rp = 1 - 2 * np.var(p_values_wilcoxon)

	    overall_Re_for_each_bias_wilcoxon.append(Re)
	    overall_Rp_for_each_bias_wilcoxon.append(Rp)

	        ## Paired t-test

	    overall_average_p_values_for_each_bias_paired_t_test.append(np.average(p_values_ttest))
	    overall_number_of_test_failure_for_each_bias_paired_t_test.append(test_result_ttest.count("failed"))

	    Re = ((test_result_ttest.count("pass") * (test_result_ttest.count("pass") - 1)) 
	                    + (test_result_ttest.count("failed") * (test_result_ttest.count("failed") - 1))) / (len(test_result_ttest) * (len(test_result_ttest) - 1))
	    Rp = 1 - 2 * np.var(p_values_ttest)

	    overall_Re_for_each_bias_paired_t_test.append(Re)
	    overall_Rp_for_each_bias_paired_t_test.append(Rp)

	        ## standard McNemar's Test

	    overall_average_p_values_for_each_bias_exact_McNemar.append(np.average(p_value_mcnemar_std))
	    overall_number_of_test_failure_for_each_bias_exact_McNemar.append(test_result_mcnemar_std.count("failed"))

	    Re = ((test_result_mcnemar_std.count("pass") * (test_result_mcnemar_std.count("pass") - 1)) 
	                    + (test_result_mcnemar_std.count("failed") * (test_result_mcnemar_std.count("failed") - 1))) / (len(test_result_mcnemar_std) * (len(test_result_mcnemar_std) - 1))
	    Rp = 1 - 2 * np.var(p_value_mcnemar_std)

	    overall_Re_for_each_bias_exact_McNemar.append(Re)
	    overall_Rp_for_each_bias_exact_McNemar.append(Rp)

	        ## Friedman's test

	    overall_average_p_values_for_each_bias_Friedman.append(np.average(p_value_Friedman))
	    overall_number_of_test_failure_for_each_bias_Friedman.append(test_result_Friedman.count("failed"))   

	    Re = ((test_result_Friedman.count("pass") * (test_result_Friedman.count("pass") - 1)) 
	                    + (test_result_Friedman.count("failed") * (test_result_Friedman.count("failed") - 1))) / (len(test_result_Friedman) * (len(test_result_Friedman) - 1))
	    Rp = 1 - 2 * np.var(p_value_Friedman)

	    overall_Rp_for_each_bias_Friedman.append(Re)   
	    overall_Re_for_each_bias_Friedman.append(Rp)

	        ## Overall Post-hoc test Results

	    overall_number_of_test_failure_for_each_bias_Nemenyi.append(test_result_nemenyi.count("failed"))
	    overall_number_of_test_failure_for_each_bias_Holm.append(test_result_holm.count("failed"))
	    overall_number_of_test_failure_for_each_bias_Shaffer.append(test_result_shaffer.count("failed"))
	    overall_number_of_test_failure_for_each_bias_Unadjusted.append(test_result_unadjusted.count("failed"))

	return  overall_average_p_values_for_each_bias_wilcoxon, overall_number_of_test_failure_for_each_bias_wilcoxon, \
	overall_Rp_for_each_bias_wilcoxon, overall_Re_for_each_bias_wilcoxon, overall_average_p_values_for_each_bias_paired_t_test, \
	overall_number_of_test_failure_for_each_bias_paired_t_test, overall_Rp_for_each_bias_paired_t_test,\
	overall_Re_for_each_bias_paired_t_test, overall_average_p_values_for_each_bias_exact_McNemar,\
	overall_number_of_test_failure_for_each_bias_exact_McNemar, overall_Rp_for_each_bias_exact_McNemar,\
	overall_Re_for_each_bias_exact_McNemar, overall_average_p_values_for_each_bias_asym_McNemar, \
	overall_number_of_test_failure_for_each_bias_asym_McNemar, overall_Rp_for_each_bias_asym_McNemar,\
	overall_Re_for_each_bias_asym_McNemar, overall_average_p_values_for_each_bias_Friedman, \
	overall_number_of_test_failure_for_each_bias_Friedman, overall_Rp_for_each_bias_Friedman, \
	overall_Re_for_each_bias_Friedman, overall_number_of_test_failure_for_each_bias_Nemenyi, \
	overall_number_of_test_failure_for_each_bias_Holm, overall_number_of_test_failure_for_each_bias_Shaffer,\
	overall_number_of_test_failure_for_each_bias_Unadjusted