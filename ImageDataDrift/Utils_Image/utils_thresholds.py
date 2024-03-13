import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils_DriftTest import access_components_kdim
from utils_DriftTest import univariate_tests
from utils_DriftTest import mmd_rbf
from utils_dimRedDef import reducer_PCA_fit
from utils_dimRedDef import reducer_umap_fit
from utils_dimRedDef import U_autoencoder_init
from utils_dimRedDef import T_autoencoder_init
from utils_dimRedDef import scale_dataset
from utils_dimRedDef import init_scaler


def get_batches (seed, df, nBatches = 5):
    ''' from df get list of batches. 
    seed: seed for shuffle dataframe
    df: input dataframe
    nBatches: number of batches to divide df into'''
    shuffled = df.sample(random_state=seed, frac=1)                # shuffle dataset- frac is fraction of rows to shuffle
    batches = np.array_split(shuffled, nBatches)       # is a list with nBatches df; here nBatches=5 empirically defined
    return batches

def permutation_list(batches):
    '''get list of couples from permutation on batches: each portion of the original ds must be compared with all the other arrays in the split
    batches: list of dataframes we want to permute in couples.'''
    paired = [(a,b) for idx, a in enumerate(batches) for b in batches[idx+1:]]      #list of permutations of batches on 2 (10 for 5 batches, 45 for 10 batches)
    return paired


def test_on_permutation_kdim(seed_metrics, paired, t_test, ks_test, kl_div, MMD, dim_red, k):
    '''perfrom tests on permutation and save results in the lists given as input
    seed_metrics
    seed_metrics: seed to set random state for tests and dimensionality reductions
    paired: list where each element is a pair of batches. Each batch is a dataframe
    t_test: list with k empty list. We will store here the results of t-test between the batches for each component
    ks_test: list with k empty list. We will store here the results of KS-test between the batches for each component
    kl_div: list with k empty list. We will store here the results of KL divergence between the batches for each component
    MMD: empty list. We will store here the results of MMD between the batches
    dim_red: string with the name of the dimensionality reduction we have applied to the batches
    k: number of dimensionalities we have reduced the batches to '''

    for pair in paired:        
        f = pair[0]         #two datasets
        g = pair[1]  


        for i in range(k):       # i is dimension we are considering, apply tests on all dimensions
            if dim_red == 'NO_dim_red': # interested only in MMD for this dimensionality reduction, sp we set by default the results of the univariate tests
                t_test[i].append(1) # since t_test and ks_test are p-value we set their result as the most conservative value, that is 1 (the two distributions are equal)
                ks_test[i].append(1)
                kl_div[i].append(0) # since kl_div is a divergence we set the default value to 0 (the two distributions are equal)
            else:
                # since after dimensionality reduction I get different structures from PCA, UMAP and autoencoder I need a function to get the k components from each dataset,
                # depending on the type of dimensionality reductor used
                f_components = access_components_kdim(f, dim_red, k)        #list of components components 
                g_components = access_components_kdim(g, dim_red, k)   

                c_X_test= f_components[i]
                c_X_val = g_components[i]

                ttest_res_val, ks_res_val, kl_res_val = univariate_tests(seed_metrics,c_X_test, c_X_val)
                t_test[i].append(ttest_res_val)
                ks_test[i].append(ks_res_val)
                kl_div[i].append(kl_res_val)            
            
        MMD_res_val = [mmd_rbf(f, g)]
        MMD.append(MMD_res_val)  

def permute_and_test(batches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k):
    '''Function for, giving the batches with dimensionality reduction techniques already applied on them, compute tests between each couple of batches.
    batches: list of dataframes. Each dataframe is a batch.
    seed_metrics: seed for reproducibility of the metrics. Is used the same for dimensionality reduction and tests.
    t_test: list of k empty lists. In each list we are going to save the results of the t-tests between the corresponding component of the batches.
    ks_test: list of k empty lists. In each list we are going to save the results of the KS-tests between the corresponding component of the batches.
    kl_div: list of k empty lists. In each list we are going to save the results of the KL-divergence between the corresponding component of the batches.
    MMD: empty list. we are going to save here the results of the MMD between each pair of batches.
    dim_red: string with name of dimensionality reduction technique already applied on each batch.
    k: number of components each batch was reduced to.'''

    paired = permutation_list(batches) # a list of couples composed by 2 different batches is returned. Each batch in the couple will be used as independent ds in the comparison to set optimised thresholds

    # Apply univariate and MMD tests on each pair in the list from permutation_list and save the results in the predefined lists
    test_on_permutation_kdim(seed_metrics, paired, t_test, ks_test, kl_div, MMD, dim_red=dim_red, k=k)  

def print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5):
    '''print thresholds as percentile from given lists
    seeds: list of seeds for fixing random state. We need them to be printed for reproducibility
    t_test: list of lists with all the results from t-tests on batches
    ks_test: list of lists with all the results from KS-test on batches
    kl_div: list of lists with all the results form KL-divergence on batches
    MMD: list with all the results from MMD on batches
    fileName: directort of the file on which print thresholds
    dim_red: string with type of the used dimensionality reductor
    percentile: we are setting the thresholds as percentile of each results' list. The default value is 5'''

    with open(fileName,'a', newline='') as file:
        file.write(f'\n Threshold for dim red {dim_red}, getted with seeds = {seeds}, for percentile = {percentile}, on 100 permutations \n')
        k = len(t_test)
        string_ttest, string_kstest, string_kldiv = '', '', ''          #initialize empty string
        for i in range(k):
            string_ttest += str(np.percentile(t_test[i], percentile))+','        # p-values so we take 5-th percentile
            string_kstest += str(np.percentile(ks_test[i], percentile))+','
            string_kldiv += str(np.percentile(kl_div[i], 100-percentile))+','    # divergence so we take 95-th percentile

        file.write('t-test \n')
        file.write(f'{string_ttest} \n')
        
        file.write('ks-test \n')
        file.write(f'{string_kstest} \n')

        file.write('kl_div \n')
        file.write(f'{string_kldiv} \n')
        
        file.write('MMD \n')
        file.write(str(np.percentile(MMD, 100-percentile))+'\n')                # discrepansy so we take 95-th percentile

        file.close()

def thresholds_None(seeds, df,  fileName, k, n_seeds=10, nBatches = 10):
    '''function to get thresholds for No_dim_red: it performs tests on divided and permuted df with fixed seed and
    prints percentile thresholds on fileName. For univariate tests we have a threshold for each component.
    seeds: list of seeds for reproducibility
    df: dataframe on which getting the thresholds
    fileName: directory of the file on which printing the thresholds
    k: number of dimenions used in the metrics.
    n_seeds: how many diffrent seeds we want to use for generate different batches. Default is 10.
    nBatches: how many batches divide our dataframe into. Default is 10.'''

    [seed_split, _, seed_metrics] = seeds
    t_test, ks_test, kl_div, MMD = [], [], [], []   # where we store our tests result
    for i in range(k):      # create an empty list for each dimension
        t_test.append([])
        ks_test.append([])
        kl_div.append([])
    dim_red = 'NO_dim_red'
    for i in range(n_seeds):         # try 10 different ways to divide into batches
        batches = get_batches(seed_split+i,df, nBatches) 
        permute_and_test(batches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k)

    # Calculate thresholds based on 5th and 95th percentiles
    print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5)


def thresholds_PCA_images(seeds, df, fileName, k=2, n_seeds=10, nBatches = 10):
    '''function to get thresholds for PCA: it performs tests on divided and permuted df after PCA with fixed seed and
    prints 5 percentile thresholds on fileName.
    seeds: list of seeds for reproducibility
    df: dataframe on which getting the thresholds
    fileName: directory of the file on which printing the thresholds
    k: number of dimenions used in the metrics.
    n_seeds: how many diffrent seeds we want to use for generate different batches. Default is 10.
    nBatches: how many batches divide our dataframe into. Default is 10.'''

    [seed_split, _, seed_metrics] = seeds
    t_test, ks_test, kl_div, MMD = [], [], [], []
    for i in range(k):
        t_test.append([])
        ks_test.append([])
        kl_div.append([])
    dim_red = 'PCA'
    for j in range(n_seeds):         # in the end I have 100 values
        batches = get_batches(seed_split+j,df, nBatches)
        # Initialize the PCA reducer and fit it on the first df in the batches (a df is needed)
        pca = reducer_PCA_fit(batches[0], k)
        # Apply PCA reduction on all the df in batches
        pcbatches = [pca.transform(batch) for batch in batches]
        # Same process as thresholds_None but on reduced batches
        permute_and_test(pcbatches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k)

    print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5)

def thresholds_UMAP_images(seeds, df, fileName, k=2, n_seeds=10, nBatches = 10):
    '''function to get thresholds for UMAP: it performs tests on divided and permuted df after UMAP with fixed seed and
    prints 5 percentile thresholds on fileName
    seeds: list of seeds for reproducibility
    df: dataframe on which getting the thresholds
    fileName: directory of the file on which printing the thresholds
    k: number of dimenions used in the metrics.
    n_seeds: how many diffrent seeds we want to use for generate different batches. Default is 10.
    nBatches: how many batches divide our dataframe into. Default is 10.'''
    
    [seed_split, _, seed_metrics] = seeds
    
    t_test, ks_test, kl_div, MMD = [], [], [], []
    for i in range(k):
        t_test.append([])
        ks_test.append([])
        kl_div.append([])
    dim_red = 'UMAP'
    for j in range(n_seeds):         # in the end I have 100 values
        batches = get_batches(seed_split+j,df, nBatches)
        # Initialize UMAP reducer and fit it on the first df in the batches (a df is needed)
        umap = reducer_umap_fit(seed_metrics, batches[0], k)
        # Apply UMAP reduction on all the df in batches
        umap_batches = [umap.transform(batch) for batch in batches]
        # Same process as thresholds_None but on reduced batches
        permute_and_test(umap_batches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k)

    print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5)

def split_dataset(df, seed):
    '''Split df in X_train and X_test
    df: dataframe we want to split.
    seed: int value used to set the random state for splitting in reproducible way.'''
    X_train, X_test = train_test_split(df, test_size=0.6, random_state = seed)       #same metric seed, because is something we do separatly from drift and training of random forest
    return X_train, X_test


def thresholds_AUTOENCODER_U_images(seeds, df, fileName, k=2, n_seeds=10, nBatches=10):
    '''function to get thresholds for Untrained Autoencoder: it performs tests on divided and permuted df after U_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches
    seeds: list of seeds for reproducibility
    df: dataframe on which getting the thresholds
    fileName: directory of the file on which printing the thresholds
    k: number of dimenions used in the metrics.
    n_seeds: how many diffrent seeds we want to use for generate different batches. Default is 10.
    nBatches: how many batches divide our dataframe into. Default is 10.'''

    [seed_split, _, seed_metrics] = seeds
    t_test, ks_test, kl_div, MMD = [], [], [], []
    for i in range(k):
        t_test.append([])
        ks_test.append([])
        kl_div.append([])
    dim_red = 'U_AUTOENCODER'

    # initialize standard scaler to scale the batches
    X_train, _ = split_dataset(df, seed_metrics)
    standard_scaler = init_scaler(X_train)

    for j in range(n_seeds):         # in the end I have 100 values
        batches = get_batches(seed_split+j,df, nBatches)      #divide in batches

        scaled_batches = [pd.DataFrame(     # scale each batch
            standard_scaler.transform(batch),
            columns = batch.columns) 
            for batch in batches]
        # Initialize untrained autoencoder reducer and fit it on the first df in the batches (a df is needed)
        encoder = U_autoencoder_init(seed_metrics, scaled_batches[0], k)   # initialize autoencoder with first scaled batch
        # Apply untrained autoencoder reduction on all the df in batches
        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    # encode the single scaled batch
        permute_and_test(encoded_batches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k)
        
    print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5)

def thresholds_AUTOENCODER_T_images(seeds, df, fileName, k=2, n_seeds=10, nBatches = 10):
    '''function to get thresholds for Trained Autoencoder: it performs tests on divided and permuted df after T_AUTOENCODER with fixed seed and
    prints 5 percentile thresholds on fileName. Observe that for autoencoder we need scaled batches
    seeds: list of seeds for reproducibility
    df: dataframe on which getting the thresholds
    fileName: directory of the file on which printing the thresholds
    k: number of dimenions used in the metrics.
    n_seeds: how many diffrent seeds we want to use for generate different batches. Default is 10.
    nBatches: how many batches divide our dataframe into. Default is 10.'''

    [seed_split, _, seed_metrics] = seeds

    t_test, ks_test, kl_div, MMD = [], [], [], []
    for i in range(k):
        t_test.append([])
        ks_test.append([])
        kl_div.append([])
        
    dim_red = 'T_AUTOENCODER'

    # initialize standard scaler to scale the batches
    X_train, X_test = split_dataset(df, seed_metrics)
    standard_scaler = init_scaler(X_train)
    #scaling test and train
    X_train_scaled = scale_dataset(X_train, standard_scaler)
    X_test_scaled = scale_dataset(X_test, standard_scaler)


    for j in range(n_seeds):         # in the end I have 100 values
    
        # batches on just X_test: for trained autoencoder we need to define where training it. We decide to split the dataset in train and test
        # and to train on train set, while compute tests on the batches of X_test. In this way we avoid passing to the encoder data he was trained on.
        #TODO: SPIEGARE IL COMMENTO ALLA RIGA SOPRA
        batches = get_batches(seed_split+j, X_test, nBatches)      
        scaled_batches = [pd.DataFrame(     # get scaled batches
                standard_scaler.transform(batch),
                columns = batch.columns
            ) for batch in batches]
        

        encoder = T_autoencoder_init(seed_metrics, X_train_scaled, X_test_scaled, k)       #initialize and train autoencoder

        encoded_batches = [pd.DataFrame(encoder.predict(batch)) for batch in scaled_batches]    #get encoded batches
        permute_and_test(encoded_batches, seed_metrics, t_test, ks_test, kl_div, MMD, dim_red, k)       
     
    print_thresholds(seeds, t_test, ks_test, kl_div, MMD,  fileName, dim_red, percentile=5)
