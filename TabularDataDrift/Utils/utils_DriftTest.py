import math
import csv
import numpy as np
from sklearn import metrics
from scipy import stats


#KL divergence
def range01(x):
    min_x = min(x)
    max_x = max(x)
    if(min_x == max_x):
        arr = x+1
    else: 
        arr = (x-min_x)/(max_x-min_x)
    arr = arr/sum(arr)
    return arr

def KL(pk, qk, EPSILON=0.00001):

    Q = np.copy(qk)
    P = np.copy(pk)

    Q = range01(np.sort(Q))               # normalize and sum equal to 1
    P = range01(np.sort(P))
    dist = 0
    PQratio = 0
    if (len(P)!=len(Q)):
        print('KeyError: The vectors you are comparing do not have the same length!')
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
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def statistical_tests_def(test_component, val_component):
    '''ttest, ks test, kl div tests for the row components'''
    ttest_res = stats.ttest_ind(test_component, val_component, nan_policy='omit').pvalue 
    ks_res = stats.kstest(test_component, val_component).pvalue 
    kl_res = KL(test_component, val_component)
    
    return ttest_res, ks_res, kl_res


def statistical_tests_exe(c_X_test, c_X_test0, c_X_test1, c_X_val, c_X_val0, c_X_val1, c_XB_val, c_XB_val0, c_XB_val1):
    '''Apply tests on components, return results as list of results for components'''
    #   WITHOUT DRIFT
    ttest_res0_val, ks_res0_val, kl_res0_val = statistical_tests_def(c_X_test0, c_X_val0)
    ttest_res1_val, ks_res1_val, kl_res1_val = statistical_tests_def(c_X_test1, c_X_val1)
    MMD_res_val = mmd_rbf(c_X_test, c_X_val)

    val_results = [ttest_res0_val, ttest_res1_val, ks_res0_val, ks_res1_val, kl_res0_val, kl_res1_val, MMD_res_val]

    # WITH DRIFT
    ttest_res0_drift, ks_res0_drift, kl_res0_drift = statistical_tests_def(c_X_test0, c_XB_val0)
    ttest_res1_drift, ks_res1_drift, kl_res1_drift = statistical_tests_def(c_X_test1, c_XB_val1)
    MMD_res_drift = mmd_rbf(c_X_test, c_XB_val)

    drift_results = [ttest_res0_drift, ttest_res1_drift, ks_res0_drift, ks_res1_drift, kl_res0_drift, kl_res1_drift, MMD_res_drift]

    results = val_results + drift_results

    return results



def access_components(X_test):
    '''How access components after PCA and UMAP'''
    X_0 = X_test[:,0]
    X_1 = X_test[:,1]
    return  X_0, X_1

def get_components(X_test, X_val, XB_val):
    '''Get 0 and 1 component for test, validation and drifted validation. In this case are the two components after PCA and UMAP'''
    X_test0, X_test1 = access_components(X_test)
    X_val0, X_val1 = access_components(X_val)
    XB_val0, XB_val1 = access_components(XB_val)
    return X_test0, X_test1, X_val0, X_val1, XB_val0, XB_val1


def access_components_NORed(X_test):
    '''How access components with no dim reduction'''
    return  X_test.iloc[:,0], X_test.iloc[:,1]

def get_components_NORed(X_test, X_val, XB_val):
    '''Get components for X_test, X_val and XB_val where the components are the feature on which the drift is simulated'''
    X_test0, X_test1 = access_components_NORed(X_test)
    X_val0, X_val1 = access_components_NORed(X_val)
    XB_val0, XB_val1 = access_components_NORed(XB_val)
    return X_test0, X_test1, X_val0, X_val1, XB_val0, XB_val1


def access_components_autoencoder(X_test):
    '''How access components after encoder'''
    return  X_test[0], X_test[1]

def get_components_autoencoder(X_test, X_val, XB_val):
    '''Get 0 and 1 component for test, validation and drifted validation. In this case are the two components after encoder'''
    X_test0, X_test1 = access_components_autoencoder(X_test)
    X_val0, X_val1 = access_components_autoencoder(X_val)
    XB_val0, XB_val1 = access_components_autoencoder(XB_val)
    return X_test0, X_test1, X_val0, X_val1, XB_val0, XB_val1



def print_statistical_results(seed, c_X_test, c_X_val,  c_XB_val, resultFile, info_drift, dim_red, feature, language = False):
    ''' Function that performs statistical tests and print results. It needs val, test and valB as single components or dimensions and total dataset
    in order to compute tests on single components and MMD on whole data. We need single component because the way to access is different for dim_red.
    resultFile is the file where frinting results, info_notests are the info for the row without tests'''
    if(dim_red == 'PCA') or (dim_red == 'UMAP'):
        c_X_test0, c_X_test1, c_X_val0, c_X_val1, c_XB_val0, c_XB_val1 =  get_components(c_X_test, c_X_val, c_XB_val)
    elif (dim_red == 'NO_dim_red'): 
        if language:
            X_test = c_X_test.iloc[:,0:2]
            X_val = c_X_val.iloc[:,0:2]
            XB_val = c_XB_val.iloc[:,0:2]
        else: 
            print(language)
            X_test = c_X_test[feature]
            X_val = c_X_val[feature]
            XB_val = c_XB_val[feature]
        c_X_test0, c_X_test1, c_X_val0, c_X_val1, c_XB_val0, c_XB_val1 =  get_components_NORed(X_test, X_val, XB_val)
    else:
        c_X_test0, c_X_test1, c_X_val0, c_X_val1, c_XB_val0, c_XB_val1 =  get_components_autoencoder(c_X_test, c_X_val, c_XB_val)
    info_notests = info_drift + [dim_red]

    np.random.seed = seed
    np.random.RandomState(seed=seed) 
    statistical_results = statistical_tests_exe(c_X_test, c_X_test0, c_X_test1, c_X_val, c_X_val0, c_X_val1, c_XB_val, c_XB_val0, c_XB_val1)
    results = info_notests + statistical_results     
    
    with open(resultFile,'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(results)
        csvfile.close()






