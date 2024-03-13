import pandas as pd
import numpy as np
from utils_training import trainingRandomForest
from utils_training import compute_accuracy
from utils_DriftSimulation import simulateGaussianDrift
from utils_DriftSimulation import simulateLinearDrift
from utils_DriftTest import print_statistical_results
from utils_dimRedDef import scale_datasets
from utils_dimRedDef import initialize_DimReduction
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from utils_thresholds import get_batches, permutation_list, test_on_permutation, print_thresholds
from utils_dimRedDef import reducer_PCA_fit
from utils_dimRedDef import reducer_umap_fit
from utils_dimRedDef import U_autoencoder_init
from utils_dimRedDef import T_autoencoder_init
from utils_thresholds import split_and_scale
from sklearn.model_selection import train_test_split
from utils_dimRedDef import scale_single_dataset




def get_prompt(df):

    prompt_list = []
    for i in range(len(df)):
        prompt = '. '.join(['The ' + col + ' is ' + str(df[col].iloc[i]) for col in df.columns])
        prompt_list.append(prompt)
    return prompt_list

def get_embeddings_df(prompt_list, embed_model):

    emb_df = pd.DataFrame(columns=list(np.ones(768)))
    for prompt in prompt_list:
        query_result = embed_model.embed_query(prompt)
        emb_df.loc[len(emb_df)] = query_result
        
    return emb_df

def get_language_embeddings(X, embed_model):
    prompt_list_x = get_prompt(X)
    emb_x = get_embeddings_df(prompt_list_x, embed_model)

    return emb_x



def generate_gaussian_llm(df, labelCol, DATASET, feature, resultFile, seed_list):
    '''Given dataset and relative informations simulate Gaussian drift and performs tests on simulated drift. Saves result in resultFile.'''
    DRIFT = 'Gaussian'

    a = 42      #42 is a random number. Its function is to let us known that the variable a and b are fixed and are not changing as we are in Gaussian case.
    b = 42


    seed_training = seed_list[0]
    seed_drift = seed_list[1]
    seed_metrics = seed_list[2]

    embeddings_model = "sentence-transformers/all-mpnet-base-v2"
    embed_model = HuggingFaceEmbeddings(model_name=embeddings_model)

    # split dataset and train random forest for classification
    X_train, X_val, X_test, y_val, y_test, testAcc, forest = trainingRandomForest(seed_training, df, labelCol) 

    emb_X_test = get_language_embeddings(X_test, embed_model)
    emb_X_val = get_language_embeddings(X_val, embed_model)
    emb_X_train = get_language_embeddings(X_train, embed_model)

    # accuracy
    accuracy = compute_accuracy(X_val, y_val, forest)

    # scaling for autoencoder
    X_train_scaled, X_test_scaled,  X_val_scaled, standard_scaler = scale_datasets(emb_X_train, emb_X_test, emb_X_val)

    info_dataset = [DATASET, DRIFT, seed_training, seed_drift, seed_metrics, testAcc]

    #values on which parameters are varying
    deltaValues = [1, 10, 100]
    meanValues = [0.0, 10.0, 100.0, 1000.0]

    #initialize dimensionality reduction
    reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer = initialize_DimReduction(seed_metrics, emb_X_test,  X_train_scaled, X_test_scaled)




    for delta in deltaValues:
        for MEAN in meanValues:
            # here we simulate drift and perform tests 

            # Gaussian noise
            XB_val = simulateGaussianDrift(seed_drift,X_val, delta, MEAN, feature)
            # drifted accuracy
            accuracyB = compute_accuracy(XB_val, y_val, forest)
            info_drift = info_dataset + [delta, MEAN, a, b, accuracy, accuracyB]

            emb_XB_val = get_language_embeddings(XB_val, embed_model)
           
                       
            # scaling XB_val for autoencoder
            XB_val_scaled = pd.DataFrame(
                standard_scaler.transform(emb_XB_val),
                columns = emb_XB_val.columns
            )
            
            #case base
            dim_red = 'NO_dim_red'
            print_statistical_results(seed_metrics, emb_X_test, emb_X_val, emb_XB_val, resultFile, info_drift, dim_red, feature, language=True)
        
            dim_red = 'PCA'
            pc_X_test = reducer_pca.transform(emb_X_test)
            pc_XB_val = reducer_pca.transform(emb_XB_val)       #but PCA gives almost equal components
            pc_X_val = reducer_pca.transform(emb_X_val)
            print_statistical_results(seed_metrics, pc_X_test, pc_X_val,  pc_XB_val, resultFile, info_drift, dim_red, feature)

            dim_red = 'UMAP'
            umap_X_test = reducer_umap.transform(emb_X_test)
            umap_XB_val = reducer_umap.transform(emb_XB_val)
            umap_X_val = reducer_umap.transform(emb_X_val)
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


def generate_linear_llm(df, labelCol, DATASET, feature, resultFile, seed_list):
    '''Given dataset and relative informations simulate linear drift and performs tests on simulated drift. Saves result in resultFile.'''
    DRIFT = 'Linear'

    MEAN = 42      #42 is a random number. Its function is to let us known that the variable a and b are fixed and are not changing as we are in Gaussian case.
    delta = 42


    seed_training = seed_list[0]
    seed_drift = seed_list[1]
    seed_metrics = seed_list[2]

    embeddings_model = "sentence-transformers/all-mpnet-base-v2"
    embed_model = HuggingFaceEmbeddings(model_name=embeddings_model)

    # split dataset and train random forest for classification
    X_train, X_val, X_test, y_val, y_test, testAcc, forest = trainingRandomForest(seed_training, df, labelCol) 

    emb_X_test = get_language_embeddings(X_test, embed_model)
    emb_X_val = get_language_embeddings(X_val, embed_model)
    emb_X_train = get_language_embeddings(X_train, embed_model)

    # accuracy
    accuracy = compute_accuracy(X_val, y_val, forest)

    # scaling for autoencoder
    X_train_scaled, X_test_scaled,  X_val_scaled, standard_scaler = scale_datasets(emb_X_train, emb_X_test, emb_X_val)

    info_dataset = [DATASET, DRIFT, seed_training, seed_drift, seed_metrics, testAcc]

    #values on which parameters are varying
    aValues = [0.01, 0.1, 1, 10, 100]
    bValues = [-100.0, -10.0, 0.0, 10.0, 100.0]

    #initialize dimensionality reduction
    reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer = initialize_DimReduction(seed_metrics, emb_X_test,  X_train_scaled, X_test_scaled)




    for a in aValues:
        for b in bValues:
            # here we simulate drift and perform tests 

            # Gaussian noise
            XB_val = simulateLinearDrift(X_val, a, b, feature)
            # drifted accuracy
            accuracyB = compute_accuracy(XB_val, y_val, forest)
            info_drift = info_dataset + [delta, MEAN, a, b, accuracy, accuracyB]

            emb_XB_val = get_language_embeddings(XB_val, embed_model)
           
                       
            # scaling XB_val for autoencoder
            XB_val_scaled = pd.DataFrame(
                standard_scaler.transform(emb_XB_val),
                columns = emb_XB_val.columns
            )
            
            #case base
            dim_red = 'NO_dim_red'
            print_statistical_results(seed_metrics, emb_X_test, emb_X_val, emb_XB_val, resultFile, info_drift, dim_red, feature, language=True)
        
            dim_red = 'PCA'
            pc_X_test = reducer_pca.transform(emb_X_test)
            pc_XB_val = reducer_pca.transform(emb_XB_val)       #but PCA gives almost equal components
            pc_X_val = reducer_pca.transform(emb_X_val)
            print_statistical_results(seed_metrics, pc_X_test, pc_X_val,  pc_XB_val, resultFile, info_drift, dim_red, feature)

            dim_red = 'UMAP'
            umap_X_test = reducer_umap.transform(emb_X_test)
            umap_XB_val = reducer_umap.transform(emb_XB_val)
            umap_X_val = reducer_umap.transform(emb_X_val)
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


def get_emb_batches(batches):
    embeddings_model = "sentence-transformers/all-mpnet-base-v2"
    embed_model = HuggingFaceEmbeddings(model_name=embeddings_model)
    emb_batches = []
    for x in batches:
        x = get_language_embeddings(x,embed_model)
        emb_batches.append(x)

    return emb_batches


def thresholds_None_llm(seed, df, feature, fileName):
    '''function to get thresholds for No_dim_red: it performs tests on divided and permuted df with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'NO_dim_red'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)
        emb_batches = get_emb_batches(batches)
        paired = permutation_list(emb_batches)

        # emb_paired = [(get_language_embeddings(a, embed_model),get_language_embeddings(b, embed_model)) for (a,b) in permutation_list]
        test_on_permutation(seed, paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red, language = True)
        i += 1
    
    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)



def thresholds_PCA_llm(seed, df, feature, fileName):
    '''function to get thresholds for PCA: it performs tests on divided and permuted df after PCA with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'PCA'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)
        emb_batches = get_emb_batches(batches)
        pca = reducer_PCA_fit(emb_batches[0])
        pcbatches = [pca.transform(batch) for batch in emb_batches]

        pca_paired = permutation_list(pcbatches)

        test_on_permutation(seed, pca_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1
        
    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)



def thresholds_UMAP_llm(seed, df,  feature, fileName):
    '''function to get thresholds for UMAP: it performs tests on divided and permuted df after UMAP with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'UMAP'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)
        emb_batches = get_emb_batches(batches)

        umap = reducer_umap_fit(seed, emb_batches[0])
        umap_batches = [umap.transform(batch) for batch in emb_batches]

        umap_paired = permutation_list(umap_batches)

        test_on_permutation(seed, umap_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

def thresholds_AUTOENCODER_U_llm(seed, df, feature, fileName):
    '''function to get thresholds for Untrained Autoencoder: it performs tests on divided and permuted df after U_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'U_AUTOENCODER'
    for i in range(10):         # in the end I have 100 values

        # _____, _____, standard_scaler, _____ = split_and_scale(df, seed)   # to get the scaler
        batches = get_batches(seed,df)      #divide in batches
        emb_batches = get_emb_batches(batches)
        _____, _____, standard_scaler, _____ = split_and_scale(emb_batches[0], seed)   # to get the scaler
        scaled_batches = [pd.DataFrame(     # scale each batch
            standard_scaler.transform(batch),
            columns = batch.columns) 
            for batch in emb_batches]
        
        encoder = U_autoencoder_init(seed, scaled_batches[0])   # initialize autoencoder with first scaled batch

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    # encode the single scaled batch
        
        encoded_paired = permutation_list(encoded_batches)      # permutation on encoded batches

        test_on_permutation(seed, encoded_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

def thresholds_AUTOENCODER_T_on_test_llm(seed, df, feature, fileName):
    '''function to get thresholds for Trained Autoencoder: it performs tests on divided and permuted df after T_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'T_AUTOENCODER_batches on test'
    for i in range(10):         # in the end I have 100 values
        
        X_train, X_test = train_test_split(df, test_size=0.6, random_state = seed)       #same metric seed, because is something we do separatly from drift and training of random forest
        emb_train = get_emb_batches([X_train])[0]       #train embedding
        emb_test = get_emb_batches([X_test])[0]       #train embedding

        x_train_scaled, standard_scaler = scale_single_dataset(emb_train)       #scale train embedding, also get standard scaler
        x_test_scaled = pd.DataFrame(     # get scaled test embedding
                standard_scaler.transform(emb_test),
                columns = emb_test.columns
            )
        # x_train_scaled, x_test_scaled, standard_scaler, X_test = split_and_scale(df, seed)        # get scaler and scaled train and test
      

        batches = get_batches(seed, X_test)      #batches on just X_test
        emb_batches = get_emb_batches(batches)      # language embedding of batches

        scaled_batches = [pd.DataFrame(     # get scaled batches
                standard_scaler.transform(batch),
                columns = batch.columns
            ) for batch in emb_batches]
        
        encoder = T_autoencoder_init(seed, x_train_scaled, x_test_scaled)       #initialize and train autoencoder

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    #get encoded batches
        
        encoded_paired = permutation_list(encoded_batches)

        test_on_permutation(seed, encoded_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

