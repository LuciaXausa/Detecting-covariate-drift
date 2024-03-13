import pandas as pd
from utils_DriftTest import print_statistical_results_kdim
from os import listdir  
import random     
import numpy as np



def split_data(input_path, seed):
    '''From input_path take folder and return train, test and validation of images from it.
    train, test and val are a list of names. In this way we can get drifted val from drifted folder.
    input_path: directory of folder of images we want spilt in different sets.
    seed: also called split_seed, is used to set the random State for reproduce the random splitting.
    It returns three lists with names of images.'''
    
    imagesList = listdir(input_path)
    
    rs = np.random.RandomState(seed)

    random.seed=seed
    rs.shuffle(imagesList)

    len_datalist = len(imagesList)
    trainLen = int(len_datalist*0.4)
    testLen = int(len_datalist*0.3)
    train = imagesList[0:trainLen]
    test = imagesList[trainLen: trainLen+testLen]
    val = imagesList[trainLen+testLen:]
    return train,test,val



def reduced_on_drift_kdim(df,  info_dataset,  reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer, sigma = 1, drift = None):
    '''Apply dimensionality reduction to drifted val. The reduced components are saved in a list.
    Info dataset is updated with drift and sigma info.
    df: dataframe we want to reduce
    info_dataset: string with info of the dataset. We are going to print it in the end, so we add the info of the drift.
    reducer_pca: reducer initialized as PCA
    reducer_umpa: reducer initialized as UMAP
    U_encoder_layer: ecoder form untrained autoencoder to be used as dimensionality reductor
    T_encoder_layer: ecoder form trained autoencoder to be used as dimensionality reductor
    sigma: int that represents the intensity (in particular the variance) of the Gaussian noise we have simulated on the images. If the drift is not Gaussian sigma is 42
    drift: string of the name of the drift.
    Returns info_drift and val_dim_red that is a list with all the reults of dimensionality reductions'''

    if(drift != None):
        info_drift = info_dataset + [drift, sigma]
    else:
        info_drift = info_dataset + ['val', 42]

    pc_drifted = reducer_pca.transform(df)
    umap_drifted = reducer_umap.transform(df)
    reduced_drifted = pd.DataFrame(U_encoder_layer.predict(df))
    T_reduced_drifted = pd.DataFrame(T_encoder_layer.predict(df))

    val_dim_red = [pc_drifted, umap_drifted, reduced_drifted, T_reduced_drifted]

    return val_dim_red, info_drift

def test_on_reduced_kdim( X, X_test,  X_dim_red, test_dim_red, seed_metrics, resultFile, info_drift, k=2):
    '''Call tests on reduced variable and print them.
    X: target dataframe: we are going to investigate if its data are different from test data in distribution.
    X_test: test dataframe, used to perform test both with X
    X_dim_red: list with reults of X after dimensionality reduction with 4 different techniques (PCA, UMAP, U autoencoder, T autoencoder)
    test_dim_red: list with results of test after dimensionality reduction
    seed_metrics: seed used for set random state for dimensionality reductors and tests
    resultFile: csv file where print reults from tests and info for reproducibility
    info_drift: info of the dataset and type of considered drift
    k: number of dimensions to redue the dataframe to.'''
    [pc_X_test, umap_X_test, reduced_X_test, T_reduced_X_test] = test_dim_red
    [pc_X, umap_X, reduced_X, T_reduced_X] = X_dim_red

   
    dim_red = 'NO_dim_red'
    print_statistical_results_kdim(seed_metrics, X_test, X, resultFile, info_drift, dim_red, k)

    dim_red = 'PCA'
    print_statistical_results_kdim(seed_metrics, pc_X_test, pc_X, resultFile, info_drift, dim_red, k)
    
    dim_red = 'UMAP'
    print_statistical_results_kdim(seed_metrics, umap_X_test, umap_X, resultFile, info_drift, dim_red, k)

    dim_red = 'U_AUTOENCODER'
    print_statistical_results_kdim(seed_metrics, reduced_X_test, reduced_X, resultFile, info_drift, dim_red, k)

    dim_red = 'T_AUTOENCODER'
    print_statistical_results_kdim(seed_metrics, T_reduced_X_test, T_reduced_X, resultFile, info_drift, dim_red, k)