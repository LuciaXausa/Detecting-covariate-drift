import pandas as pd
from utils_training import trainingRandomForest
from utils_training import compute_accuracy
from utils_DriftSimulation import simulateGaussianDrift
from utils_DriftSimulation import simulateLinearDrift
from utils_DriftTest import print_statistical_results
from utils_dimRedDef import scale_datasets
from utils_dimRedDef import initialize_DimReduction

import numpy as np


def generate_gaussian(df, labelCol, DATASET, feature, resultFile, seed_list):
    '''Given dataset and relative informations simulate Gaussian drift and performs tests on simulated drift. Saves result in resultFile.'''
    DRIFT = 'Gaussian'

    a = 42      #42 is a random number. Its function is to let us known that the variable a and b are fixed and are not changing as we are in Gaussian case.
    b = 42


    seed_training = seed_list[0]
    seed_drift = seed_list[1]
    seed_metrics = seed_list[2]



    # split dataset and train random forest for classification
    X_train, X_val, X_test, y_val, y_test, testAcc, forest = trainingRandomForest(seed_training, df, labelCol) 

    # accuracy
    accuracy = compute_accuracy(X_val, y_val, forest)

    # scaling for autoencoder
    X_train_scaled, X_test_scaled,  X_val_scaled, standard_scaler = scale_datasets(X_train, X_test, X_val)

    info_dataset = [DATASET, DRIFT, seed_training, seed_drift, seed_metrics, testAcc]

    #values on which parameters are varying
    # # deltaValues = [1, 10, 100]
    # deltaValues = [2.5, 5, 7.5]
    # meanValues = [0.0, 10.0, 100.0, 1000.0]

    deltaValues = [1.0, 5.0, 10.0, 50,100,150]
    meanValues = [ -50,-40,-30,-20,-10.0, -5.0, 0.0, 5.0, 10.0, 20, 30, 40, 50]

    #initialize dimensionality reduction
    reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer = initialize_DimReduction(seed_metrics, X_test,  X_train_scaled, X_test_scaled)


    for delta in deltaValues:
        for MEAN in meanValues:
            # here we simulate drift and perform tests 

            # Gaussian noise
            XB_val = simulateGaussianDrift(seed_drift,X_val, delta, MEAN, feature)
            # drifted accuracy
            accuracyB = compute_accuracy(XB_val, y_val, forest)
            info_drift = info_dataset + [delta, MEAN, a, b, accuracy, accuracyB]

            # scaling XB_val for autoencoder
            XB_val_scaled = pd.DataFrame(
                standard_scaler.transform(XB_val),
                columns = XB_val.columns
            )
            
            #case base
            dim_red = 'NO_dim_red'
            print_statistical_results(seed_metrics, X_test, X_val, XB_val, resultFile, info_drift, dim_red, feature)
        
            dim_red = 'PCA'
            pc_X_test = reducer_pca.transform(X_test)
            pc_XB_val = reducer_pca.transform(XB_val)       #but PCA gives almost equal components
            pc_X_val = reducer_pca.transform(X_val)
            print_statistical_results(seed_metrics, pc_X_test, pc_X_val,  pc_XB_val, resultFile, info_drift, dim_red, feature)

            dim_red = 'UMAP'
            umap_X_test = reducer_umap.transform(X_test)
            umap_XB_val = reducer_umap.transform(XB_val)
            umap_X_val = reducer_umap.transform(X_val)
            print_statistical_results(seed_metrics, umap_X_test, umap_X_val, umap_XB_val  , resultFile, info_drift, dim_red, feature)


            dim_red = 'U_AUTOENCODER'
            reduced_X_test = pd.DataFrame(U_encoder_layer.predict(X_test_scaled))
            reduced_XB_val = pd.DataFrame(U_encoder_layer.predict(XB_val_scaled))
            reduced_X_val = pd.DataFrame(U_encoder_layer.predict(X_val_scaled))
            print_statistical_results(seed_metrics, reduced_X_test, reduced_X_val, reduced_XB_val , resultFile, info_drift, dim_red, feature)


            dim_red = 'T_AUTOENCODER'
            T_reduced_X_test = pd.DataFrame(T_encoder_layer.predict(X_test_scaled))
            T_reduced_XB_val = pd.DataFrame(T_encoder_layer.predict(XB_val_scaled))
            T_reduced_X_val = pd.DataFrame(T_encoder_layer.predict(X_val_scaled))
            print_statistical_results(seed_metrics, T_reduced_X_test, T_reduced_X_val, T_reduced_XB_val, resultFile, info_drift, dim_red, feature)

        





def generate_linear(df, labelCol, DATASET, feature, resultFile, seed_list):
    '''Given dataset and relative informations simulate Linear drift and performs tests on simulated drift. Saves result in resultFile.'''

    DRIFT = 'Linear'

    MEAN = 42        #42 is a random number. Its function is to let us known that the variable mean and delta are fixed and are not changing as we are in Gaussian case.
    delta = 42


    seed_training = seed_list[0]
    seed_drift = seed_list[1]
    seed_metrics = seed_list[2]

    # split dataset and train random forest for classification
    X_train, X_val, X_test, y_val, y_test, testAcc, forest = trainingRandomForest(seed_training, df, labelCol)

    # accuracy
    accuracy = compute_accuracy(X_val, y_val, forest)

    # scaling for autoencoder
    X_train_scaled, X_test_scaled,  X_val_scaled, standard_scaler = scale_datasets(X_train, X_test, X_val)

    info_dataset = [DATASET, DRIFT, seed_training, seed_drift, seed_metrics, testAcc]

    # aX+b
    # aValues = [0.01, 0.1, 1.0, 10.0, 100.0]
    aValues = [0.05, 0.1, 0.25, 0.5, 0.75,1, 2.5,5,7.5,10,15, 20, 30, 40, 50]
    bValues =  [ -50,-40,-30,-20,-10.0, -5.0, 0.0, 5.0, 10.0, 20, 30, 40, 50]
    #initialize dimensionality reduction
    reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer = initialize_DimReduction(seed_metrics, X_test,  X_train_scaled, X_test_scaled)


    for a in aValues:
        for b in bValues:
            # Linear drift
            XB_val = simulateLinearDrift(X_val, a, b, feature)
            # drifted accuracy
            accuracyB = compute_accuracy(XB_val, y_val, forest)
            info_drift = info_dataset + [delta, MEAN, a, b, accuracy, accuracyB]

            # scaling XB_val for autoencoder
            XB_val_scaled = pd.DataFrame(
                standard_scaler.transform(XB_val),
                columns = XB_val.columns
            )
            
            #case base
            dim_red = 'NO_dim_red'
            print_statistical_results(seed_metrics, X_test, X_val, XB_val, resultFile, info_drift, dim_red, feature)
        
            dim_red = 'PCA'
            pc_X_test = reducer_pca.transform(X_test)
            pc_XB_val = reducer_pca.transform(XB_val)       #but PCA gives almost equal components
            pc_X_val = reducer_pca.transform(X_val)
            print_statistical_results(seed_metrics, pc_X_test, pc_X_val,  pc_XB_val, resultFile, info_drift, dim_red, feature)

            dim_red = 'UMAP'
            umap_X_test = reducer_umap.transform(X_test)
            umap_XB_val = reducer_umap.transform(XB_val)
            umap_X_val = reducer_umap.transform(X_val)
            print_statistical_results(seed_metrics, umap_X_test, umap_X_val, umap_XB_val  , resultFile, info_drift, dim_red, feature)


            dim_red = 'U_AUTOENCODER'
            reduced_X_test = pd.DataFrame(U_encoder_layer.predict(X_test_scaled))
            reduced_XB_val = pd.DataFrame(U_encoder_layer.predict(XB_val_scaled))
            reduced_X_val = pd.DataFrame(U_encoder_layer.predict(X_val_scaled))
            print_statistical_results(seed_metrics, reduced_X_test, reduced_X_val, reduced_XB_val , resultFile, info_drift, dim_red, feature)


            dim_red = 'T_AUTOENCODER'
            T_reduced_X_test = pd.DataFrame(T_encoder_layer.predict(X_test_scaled))
            T_reduced_XB_val = pd.DataFrame(T_encoder_layer.predict(XB_val_scaled))
            T_reduced_X_val = pd.DataFrame(T_encoder_layer.predict(X_val_scaled))
            print_statistical_results(seed_metrics, T_reduced_X_test, T_reduced_X_val, T_reduced_XB_val, resultFile, info_drift, dim_red, feature)

            