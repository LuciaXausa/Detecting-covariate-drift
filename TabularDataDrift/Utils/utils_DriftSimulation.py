import numpy as np


def simulateGaussianDrift(seed, X_val, delta, MEAN, feature):
    '''Simulate Gaussian drift on selected features of X_val, returns drifted val'''

    rs = np.random.RandomState(seed)
    #HYPERPARAMETERS:
    #       BIVARIATE
    COV = [[delta*12,0],[ 0,delta]]     #il fattore 12 è perchè la varianza del primo feature è molto maggiore
    mean2d = [MEAN, MEAN]

    # shift Val
    XB_val = X_val.copy()
    XB_val[feature] += rs.multivariate_normal(
        mean = mean2d,
        cov = COV,
        )

    return XB_val  


def simulateLinearDrift(X_val, a, b, feature):    
    '''Simulate Linear drift on selected features of X_val, returns drifted val'''

    # shift Val
    XB_val = X_val.copy()
    XB_val[feature] = a*XB_val[feature] + b

    return XB_val 



def to_mmolL(x): 
    return x*0.0884
    
def to_decimal(x): 
    return x/100


def simulateMeasureDrift(X_val, y_val, forest, a, b):
    '''Simulate drift as change of measurement units on selected features of X_val, returns drifted val. 
    Is not used beacause it can be included in linear case.'''

    # shift Val
    XB_val = X_val.copy()
    XB_val['serum_creatinine'] = to_mmolL(XB_val['serum_creatinine'])
    XB_val['ejection_fraction'] = to_decimal(XB_val['ejection_fraction'])

    return XB_val 



