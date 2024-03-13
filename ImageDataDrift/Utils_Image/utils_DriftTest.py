import math
import csv
import numpy as np
from sklearn import metrics
from scipy import stats


# for KL divergence def
def range01(x):
    '''Scaler
    x: numpy array to be scaled'''
    min_x = min(x)
    max_x = max(x)
    if(min_x == max_x):
        arr = x+1
    else: 
        arr = (x-min_x)/(max_x-min_x)
    arr = arr/sum(arr)
    return arr

def KL(pk, qk, EPSILON=0.00001):
    '''Kullback-Leibler divergence
    pk, qk: two array of which we want to compute the KL divergence
    EPSILON: value to deal with exception given by 0/0. Default is 0.00001'''
    Q = np.copy(qk)
    P = np.copy(pk)

    Q = range01(np.sort(Q))               # normalize and sum equal to 1
    P = range01(np.sort(P))
    dist = 0
    PQratio = 0
    if (len(P)!=len(Q)):
       minLen = min(len(P), len(Q))
       P = P[0:minLen]
       Q = Q[0: minLen]
    for i in range (len(P)):
        if (P[i]==0 and Q[i]==0):
            dist += 0
        else:
            if(Q[i]==0):
                PQratio = P[i]/EPSILON
            else:
                PQratio = P[i]/Q[i]
            if(PQratio==0):
                dist += 0
            else: 
                dist += P[i]*math.log2(PQratio)
    return dist


# mmd definition
def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    X,Y: dataframes we want to compute MMD on
    gamma: int, used in the above formula
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def univariate_tests(seed_metrics, test_component, val_component):
    '''ttest, ks test, kl div tests for given components. 
    test_components, val_components: vectors we want to compute univariate tests on.
    seed_metric: int defining seed to set the random state for reproducibility of tests'''
    rng = np.random.RandomState(seed_metrics)
    ttest_res = stats.ttest_ind(test_component, val_component, nan_policy='omit', random_state=rng).pvalue 
    ks_res = stats.kstest(test_component, val_component).pvalue 
    kl_res = KL(test_component, val_component)
    
    return ttest_res, ks_res, kl_res


def statistical_tests_exe_kdim(X_test, test_components, X_val, val_components, dim_red, seed_metrics):
    '''Apply tests on components, return results as list of results for components. The order in the list is ttest0, ttest1, ..., ttestk, kstest0, ..., kldiv0, ..., MMD
    Indeed they are ready for be printed in a csv file.
    X_test, X_val: dataframes (obtained after dimensionality reduction) we want to compute tests on. Used to compute MMD since is the only multivariate test.
    test_components, val_components: list of components as vectors. Used in order to compute univariate tests between them.
    dim_red: string name of used dimensionality reduction technique. 
    seed_metrics: int value, used to set the random state of tests.
    '''
    
    result_for_dim = []
    ttest = []
    kstest = []
    kldiv = []

    for i in range(len(test_components)):       # i is dimension we are considering, apply tests on all dimensions
        if dim_red == 'NO_dim_red':
            ttest.append(1)
            kstest.append(1)
            kldiv.append(0)
        else:
            c_X_test= test_components[i]            # consider i-th component
            c_X_val = val_components[i]

            ttest_res_val, ks_res_val, kl_res_val = univariate_tests(seed_metrics, c_X_test, c_X_val)       #apply univariate tests on i-th component as comparison between test and val
            ttest.append(ttest_res_val)
            kstest.append(ks_res_val)
            kldiv.append(kl_res_val)
            
    MMD_res_val = [mmd_rbf(X_test, X_val)]      # apply MMD as multivariate test
    result_for_dim = (ttest+kstest+kldiv+MMD_res_val)

    return result_for_dim


def access_components_kdim(X, dim_red, k):
    '''How access components after different dimensionality reductions. Returns list of components.
    X: the reduced dataframe of which we want to access the components.
    k: the number of components/dimensions of X '''
    X_components = []
    for i in range(k):
        if (dim_red == 'PCA' or dim_red == 'UMAP'):
            X_components.append(X[:,i])
        else:       #autoencoder
            X_components.append(X[i])
    return  X_components

def get_components_kdim(X_test, X, dim_red, k):
    '''Get k components for test and X. 
    X_test: reduced dataframe of which we want to extract components
    X: reduced dataframe of which we want to extract components. Tests will be applied to see if X_test is statistically different from X.
    dim_red: string with the name of the used dimensionality reduction technique
    k: number of columns of dataframe, so number of components we will get.'''
    test_components = access_components_kdim(X_test, dim_red, k)
    X_components = access_components_kdim(X, dim_red, k)
    return test_components, X_components


def print_statistical_results_kdim(seed, c_X_test, c_X_val, resultFile, info_drift, dim_red, k):
    ''' Function that performs statistical tests and print results. It needs val and test after any type of dimensionality reductionin order to compute tests 
    on single components and MMD on whole data. 
    seed: also called seed_metrics, is the seed we use for set the random State of dimensionality reduction techniques and of the tests.
    resultFile: the file where printing results,
    info_notests: the info for reproducibility,
    dim_red: string with name of dimensionality reduction that was applied to our data.
    k: number of dimensions the dataframe was reduced to.'''
    test_components, val_components =  get_components_kdim(c_X_test, c_X_val, dim_red, k)
    info_notests = info_drift + [dim_red]

    statistical_results = statistical_tests_exe_kdim(c_X_test, test_components, c_X_val, val_components, dim_red, seed_metrics=seed)
    results = info_notests + statistical_results     
    
    with open(resultFile,'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(results)
        csvfile.close()

