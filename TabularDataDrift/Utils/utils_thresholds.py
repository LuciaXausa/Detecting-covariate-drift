import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils_DriftTest import KL
from utils_DriftTest import mmd_rbf
from utils_dimRedDef import reducer_PCA_fit
from utils_dimRedDef import reducer_umap_fit
from utils_dimRedDef import U_autoencoder_init
from utils_dimRedDef import T_autoencoder_init
from utils_dimRedDef import scale_datasets

from utils_DriftTest import access_components_autoencoder
from utils_DriftTest import access_components
from utils_DriftTest import statistical_tests_def

def get_batches ( seed, df, nBatches = 5):
    ''' from df get list of batches'''
    shuffled = df.sample(random_state=seed, frac=1)                # shuffle dataset
    batches = np.array_split(shuffled, nBatches)       # is a list
    return batches

def permutation_list(batches):
    '''get list of couples from permutation on batfches'''
    paired = [(a,b) for idx, a in enumerate(batches) for b in batches[idx+1:]]      #list of permutations of batches on 2 (10 for 5 batches, 45 for 10 batches)7
    return paired

def test_on_permutation(seed,paired,  ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red, language=False):
    '''perfrom tests on permutation and save results in the lists given as input'''
    np.random.seed = seed
    for pair in paired:        
        f = pair[0]
        g = pair[1]     
        if(dim_red=='NO_dim_red'):
            if(not f.size==g.size):      #g has one item less
                f = f.iloc[0:-1]            # we don't take the last one 
            if(language):           #if we are in llm case we don't have fatures but over 700 columns Where do we perform tests? on the first two columns
                f_0, f_1 = f.iloc[:,0], f.iloc[:,1]   
                g_0, g_1 = g.iloc[:,0], g.iloc[:,1]   
            else:
                f_0, f_1 = f[feature[0]], f[feature[1]]
                g_0, g_1 = g[feature[0]], g[feature[1]]
        elif((dim_red=='PCA') or (dim_red=='UMAP')):                        #((dim_red=='PCA') or (dim_red=='UMAP')):
            if(not f.size==g.size):      #g has one item less
                f = f[0:-1]            # we don't take the last one 
            f_0, f_1 = access_components(f)
            g_0, g_1 = access_components(g)
        else:
            if(not f.size==g.size):      #g has one item less
                f = f[0:-1]            # we don't take the last one 
            f_0, f_1 = access_components_autoencoder(f)
            g_0, g_1 = access_components_autoencoder(g)
        
        
        tres1, ksres1, klres1 = statistical_tests_def(f_0, g_0)
        tres2, ksres2, klres2 = statistical_tests_def(f_1, g_1)

        MMDres = mmd_rbf(f, g)
        
        ttest_0.append(tres1)
        ttest_1.append(tres2) 

        kstest_0.append(ksres1)
        kstest_1.append(ksres2)       

        kldiv_0.append(klres1)
        kldiv_1.append(klres2)

        MMD.append(MMDres)                  
            
def print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, fileName, dim_red, percentile = 5):
    '''print thresholds as percentile from given lists'''
    with open(fileName,'a', newline='') as file:
        file.write(f'Threshold for dim red {dim_red}, getted with seed = {seed}, for percentile = {percentile}, on 100 permutations \n')
        file.write(f'ttest_0 : {np.percentile(ttest_0,percentile)} \n')
        file.write(f'ttest_1 : {np.percentile(ttest_1,percentile)}\n')
        file.write(f'ksest_0 : {np.percentile(kstest_0,percentile)}\n')
        file.write(f'ksest_1 : {np.percentile(kstest_1,percentile)}\n')
        file.write(f'kldiv_0 : {np.percentile(kldiv_0,100-percentile)}\n')
        file.write(f'kldiv_1 : {np.percentile(kldiv_1,100-percentile)}\n')
        file.write(f'MMD: {np.percentile(MMD,100-percentile)}\n')
        file.close()

def thresholds_None(seed, df, feature, fileName):
    '''function to get thresholds for No_dim_red: it performs tests on divided and permuted df with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'NO_dim_red'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)
        paired = permutation_list(batches)

        test_on_permutation(seed, paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1
    
    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)



def thresholds_PCA(seed, df, feature, fileName):
    '''function to get thresholds for PCA: it performs tests on divided and permuted df after PCA with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'PCA'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)

        pca = reducer_PCA_fit(batches[0])
        pcbatches = [pca.transform(batch) for batch in batches]

        pca_paired = permutation_list(pcbatches)

        test_on_permutation(seed, pca_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)


def thresholds_UMAP(seed, df,  feature, fileName):
    '''function to get thresholds for UMAP: it performs tests on divided and permuted df after UMAP with fixed seed and
    prints 5 percentile thresholds on fileName'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'UMAP'
    for i in range(10):         # in the end I have 100 values
        batches = get_batches(seed,df)

        umap = reducer_umap_fit(seed, batches[0])
        umap_batches = [umap.transform(batch) for batch in batches]

        umap_paired = permutation_list(umap_batches)

        test_on_permutation(seed, umap_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

def split_and_scale(df, seed):
    '''Split df in X_train and X_test and returns train and test scaled and the scaler'''
    X_train, X_test = train_test_split(df, test_size=0.6, random_state = seed)       #same metric seed, because is something we do separatly from drift and training of random forest
    x_train_scaled, x_test_scaled,  ______, standard_scaler = scale_datasets(X_train, X_test, X_test)
    return x_train_scaled, x_test_scaled, standard_scaler, X_test 

def thresholds_AUTOENCODER_U(seed, df, feature, fileName):
    '''function to get thresholds for Untrained Autoencoder: it performs tests on divided and permuted df after U_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'U_AUTOENCODER'
    for i in range(10):         # in the end I have 100 values

        _____, _____, standard_scaler, _____ = split_and_scale(df, seed)   # to get the scaler
        batches = get_batches(seed,df)      #divide in batches
        scaled_batches = [pd.DataFrame(     # scale each batch
            standard_scaler.transform(batch),
            columns = batch.columns) 
            for batch in batches]
        
        encoder = U_autoencoder_init(seed, scaled_batches[0])   # initialize autoencoder with first scaled batch

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    # encode the single scaled batch
        
        encoded_paired = permutation_list(encoded_batches)      # permutation on encoded batches

        test_on_permutation(seed, encoded_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

def thresholds_AUTOENCODER_T(seed, df, feature, fileName):
    '''function to get thresholds for Trained Autoencoder: it performs tests on divided and permuted df after T_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'T_AUTOENCODER'
    for i in range(10):         # in the end I have 100 values
        
        x_train_scaled, x_test_scaled, standard_scaler, _____ = split_and_scale(df, seed)        # get scaler and scaled train and test
      

        batches = get_batches(seed,df)      #batches on whole dataset
        scaled_batches = [pd.DataFrame(     # get scaled batches
                standard_scaler.transform(batch),
                columns = batch.columns
            ) for batch in batches]
        
        encoder = T_autoencoder_init(seed, x_train_scaled, x_test_scaled)       #initialize and train autoencoder

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    #get encoded batches
        
        encoded_paired = permutation_list(encoded_batches)

        test_on_permutation(seed, encoded_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

def thresholds_AUTOENCODER_T_on_test(seed, df, feature, fileName):
    '''function to get thresholds for Trained Autoencoder: it performs tests on divided and permuted df after T_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches'''
    ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD = [], [], [], [], [], [], []
    dim_red = 'T_AUTOENCODER_batches on test'
    for i in range(10):         # in the end I have 100 values
        
        x_train_scaled, x_test_scaled, standard_scaler, X_test = split_and_scale(df, seed)        # get scaler and scaled train and test
      

        batches = get_batches(seed, X_test)      #batches on just X_test
        scaled_batches = [pd.DataFrame(     # get scaled batches
                standard_scaler.transform(batch),
                columns = batch.columns
            ) for batch in batches]
        
        encoder = T_autoencoder_init(seed, x_train_scaled, x_test_scaled)       #initialize and train autoencoder

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    #get encoded batches
        
        encoded_paired = permutation_list(encoded_batches)

        test_on_permutation(seed, encoded_paired, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD, feature, dim_red)
        i += 1

    print_thresholds(seed, ttest_0, ttest_1, kstest_0, kstest_1, kldiv_0, kldiv_1, MMD,  fileName, dim_red, percentile=5)

