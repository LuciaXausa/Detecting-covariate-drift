
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

labels = {'ks': 'p-value of KS test',
            'ttest': 'p-value of t-test',
            'MMD': 'MMD value',
            'kl': 'KL-divergence value',
            'ks_drift_0':'p-value of KS test on first component',
            'ks_drift_0_y': 'p-value',
            'MMD_drift':'MMD on drifted dataset',
            'MMD_drift_y':'MMD',
            'a':'a (scaling parameter)',
            'b': 'b (location parameter)',
            'mean':'mean',
            'sigma' :'sigma (variance of Gaussian drift)',
            'NO_dim_red': 'no dimensionality reduction',
            'PCA': 'PCA',
            'UMAP': 'UMAP',
            'U_AUTOENCODER': 'untrained autoencoder',
            'T_AUTOENCODER': 'trained autoencoder'
            }
labels_box = {'ks': 'KS p-value',
            'ttest': 't-test p-value',
            'MMD': 'MMD',
            'kl': 'KL',
            'a':'a',
            'b': 'b',
            'Linear' : 'linear (aX+b)',
            'Gaussian' :'gaussian',
            'mean':'mean',
            'sigma' :'sigma',
            'NO_dim_red': 'no dim. red.',
            'PCA': 'PCA',
            'UMAP': 'UMAP',
            'U_AUTOENCODER': 'untrained autoencoder',
            'T_AUTOENCODER': 'trained autoencoder'
            }

def plotDataframe(DRIFT, meanId, deltaId, test, dfs):
    '''dfs is list of dataframes, each one is the filtered df with fixed mean and delta values. (or a in b in linear case). 
    MeanID and deltaID are lists with respective values for mean and delta. Drift is bool and tells us if consider drift or val case.
    It returns a matrix and a dataframe, both with the same structure: for a row for each value taken by mean, a column for each value taken from delta,
    in each box the corrisponding value for the selected test. '''
    matrix_drift = []               # matrix with mean values of test
    counter = 0

    if(DRIFT): driftVal = '_drift'
    else: driftVal = '_val'
    for x in range(len(meanId)):        
        row = []                        # we get a row for each value taken by mean
        for y in range(len(deltaId)):       # for each value taken by delta
            if ((test == 'MMD') or (test == 'caseBase')):         # for test on MMD 
                #we take the MMD value
                val = dfs[counter][test + driftVal]
                averTest = val[val.index[0]]
            else:                       # in this case we have to do the mean of the two values
                column1 = dfs[counter][test + driftVal + '_0']
                column2 = dfs[counter][test + driftVal + '_1']
                # we make the mean of the two values
                meanCol = (column1+column2)/2
                averTest = meanCol[meanCol.index[0]]
            row.append(averTest)      #append the value to the row
            counter +=1
        matrix_drift.append(row)
    df_drift = pd.DataFrame(data = matrix_drift, index = meanId, columns = deltaId)     #create dataframe
    return df_drift, matrix_drift

def test_drift_heatmap(df, dim_red, drift = 'Gaussian', params = ['mean', 'sigma'], test = 'MMD', seed_list = [1 , 10, 100]):
    '''function to plot heatmap with drift params (mean and delta) varying and as values the mean of MMD
    or of the mean values of p-values or KL distance'''

    filtered = df[(df['drift']==drift)&(df['dim_red'] == dim_red)&(df['seed_training'] == seed_list[0])&(df['seed_drift'] == seed_list[1])&(df['seed_metrics'] == seed_list[2])]       
    gkk = filtered.groupby(params)       # group for my params (mean, delta), (a,b)
    dfs = [x[1] for x in list(gkk)]         # I have list, each element is a dataframe with fixed mean and delta. The order is: fixed mean and delta growing
    
    gmean =  filtered.groupby(params[0])      #to have all values of my parameters
    gdelta =  filtered.groupby(params[1])
    meanId = list(gmean.groups.keys())
    deltaId = list(gdelta.groups.keys())

    #get dataframes to plot
    driftDf, driftMatrix = plotDataframe(True, meanId, deltaId, test, dfs)      #in this case we need also matrix because it will give us thresholds for graph scale.
    refDf, ___ = plotDataframe(False, meanId, deltaId, test, dfs)   

    figure, (ax1,ax2) = plt.subplots (1,2, figsize = (24, 6))       #create figure for plots

    vmin=min(min(driftMatrix))
    vmax= max(max(driftMatrix))

    sn.heatmap(data = refDf, linewidths=2, annot=True, vmin=vmin, vmax= vmax, cmap="Greens", ax =ax1)
    ax1.set_title(f'{labels[test]} on non-drifted Target  after {labels[dim_red]}, seeds set to {seed_list}')
    ax1.set_xlabel(labels[params[1]])
    ax1.set_ylabel(labels[params[0]])

    sn.heatmap(data = driftDf, linewidths=2, annot=True, vmin = vmin , vmax = vmax, cmap="Greens", ax =ax2)
    ax2.set_title(f'{labels[test]} on {drift} drift with {labels[dim_red]}, seeds set to {seed_list}')
    plt.xlabel(labels[params[1]])
    plt.ylabel(labels[params[0]])

def test_boxplot(df, dim_red, drift = 'Gaussian', params = ['mean', 'sigma'], test = 'MMD', varying_seed = 'seed_metrics'):
    '''Builds boxplots for tests. Each boxplot depends on drift type, drift params, type of dim. reduction '''
    np.random.seed = 1      # set seed for ks test
    standard_seed = {'seed_training': 1,
                     'seed_drift': 10,
                     'seed_metrics': 100}
    fixed_seed = [k  for k in standard_seed.keys() if k != varying_seed]
    fixed_values = [standard_seed[k] for k in fixed_seed]

    filtered = df[(df['drift']==drift)&(df['dim_red'] == dim_red)&(df[fixed_seed[0]]==fixed_values[0]) & (df[fixed_seed[1]]==fixed_values[1])]           # filter on my dimensionality reduction and drift type
    gkk = filtered.groupby(params)       # group for my params (mean, delta), (a,b)
    dfs = [x[1] for x in list(gkk)]         # I have list, each element is a dataframe with fixed mean and delta. The order is: fixed mean and delta growing
    
    gmean =  filtered.groupby(params[0])      #to have all values of my parameters
    gdelta =  filtered.groupby(params[1])
    meanId = list(gmean.groups.keys())
    deltaId = list(gdelta.groups.keys())

    counter = 0

    fig, axes = plt.subplots(len(meanId),len(deltaId), figsize = (12*len(deltaId), 6*len(meanId)))

    # remember that I want a boxplot for each delta, mean valuee.
    for x in range(len(meanId)):
        for y in range(len(deltaId)):
                if(test == 'MMD'): 
                    dfNew = pd.DataFrame(data = dfs[counter], columns=[test+'_drift', test+'_val'])
                    #get p-value for boxplot width ks test
                    res = stats.ttest_ind(dfs[counter][test+'_drift'], dfs[counter][test+'_val'])
                    pValues = round(res.pvalue, 3)
                    yrange = (0,2.5)
                else: 
                    dfNew = pd.DataFrame(data = dfs[counter], columns=[test+'_drift_0',test+'_drift_1', test+'_val_0', test+'_val_1'])
                    yrange = (0,1.2)

                    res_1 = stats.ttest_ind(dfs[counter][test+'_drift_0'], dfs[counter][test+'_val_0'])
                    pValues_1 = round(res_1.pvalue, 3)
                    res_2 = stats.ttest_ind(dfs[counter][test+'_drift_1'], dfs[counter][test+'_val_1'])
                    pValues_2 = round(res_2.pvalue, 3)

                bplot = sn.boxplot(x='variable', y='value', data = pd.melt(dfNew), ax = axes[x,y])
                axes[x,y].set_title(f'{labels_box[test]} with {labels_box[dim_red]} for {labels_box[params[0]]} equal to {meanId[x]} and {labels_box[params[1]]} equal to {deltaId[y]}')
                axes[x,y].set_ylim(yrange)

                # for graphical purpose
                if(test == 'MMD'):
                    x1,x2 = 0, 1    
                    y, h, col = pd.melt(dfNew)['value'].max()+0.1, 0.1, 'k'
                    bplot.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw = 1.5, c=col)
                    bplot.text((x1+x2)*.5, y+h, f'p-value : {pValues}', ha='center', va='bottom', color=col)
                else:
                    x1,x2, x3, x4 = 0, 2, 1, 3
                    y, h, col = pd.melt(dfNew)['value'].max()+0.01, 0.01, 'k'
                    bplot.plot([x1,x1,x2,x2],[y,y+h,y+h,y], lw = 1.5, c=col)
                    bplot.text((x1+x2)*.5, y+h, f'p-value : {pValues_1}', ha='center', va='bottom', color=col)
                    bplot.plot([x3,x3,x4,x4],[y,y+7*h,y+7*h,y], lw = 1.5, c=col)
                    bplot.text((x3+x4)*.5, y+7*h, f'p-value : {pValues_2}', ha='center', va='bottom', color=col)
                        
         
                # axes[x,y].set_yscale('log')
                counter +=1
    fig.suptitle(f'{labels[test]} at varying of {varying_seed}')
    # plt.title(f'{test} at varying of {varying_seed}')
    plt.show()
    # fig.savefig(f'boxplot_{varying_seed}.png')


def test_dimRed_plot(df, fixed_param, fixed_value, changing_param, drift = 'Linear', test = 'MMD', seed_list = [1,10,100], xlim= (-50,50), ylim = (0,2)):
    '''function to plot test for varying of a param and for different dim_red. Confront with base case?'''

    filtered = df[(df['drift']==drift)&(df[fixed_param] == fixed_value)&(df['seed_training'] == seed_list[0])&(df['seed_drift'] == seed_list[1])&(df['seed_metrics'] == seed_list[2])]           # filter on my dimensionality reduction and drift type
    gkk = filtered.groupby('dim_red')       # group for my params (mean, delta), (a,b)
    dfs = [x[1] for x in list(gkk)]         # I have list, each element is a dataframe with fixed mean and delta. The order is: fixed mean and delta growing
    
    dimRed_Id = list(gkk.groups.keys())
    counter = 0
    testName ='someErrors'
    for dimRed in dfs:
        if(test == 'MMD'): 
            testName = test+'_drift'
        else: 
           testName = test+'_drift_0'
       
        # if(test == 'KL_div'):  yrange = (0,1)   
        # else: yrange = (0,1)
        plt.plot(dimRed[changing_param], dimRed[testName], label = labels[dimRed_Id[counter]], marker = 'o')
        counter+=1
    plt.title(f'{labels[test]} at varying of {changing_param} while {fixed_param} fixed to {fixed_value}')
    plt.xlabel(labels[changing_param])
    plt.ylabel(labels[test])
    # 193 plt.ylabel(labels[testName+'_y'])
    # plt.xscale('log')
    plt.ylim(ylim)
    plt.xlim(xlim)
    # plt.yscale('log')
    plt.legend()
    plt.show()


def get_matrix_drift(meanId,deltaId, dfs, DRIFT, DIFF = False):
        '''Get matrix for monitoring accuracy with respect to drift intensity'''
        matrix_drift = []               # matrix with mean values of test
        counter = 0

        if(DRIFT): driftVal = '_valB'
        else: driftVal = '_val'

        for x in range(len(meanId)):
            row = []
            for y in range(len(deltaId)):
                if DIFF: # case we want  difference of accuracy
                    val = abs(dfs[counter]['accuracy' + driftVal]-dfs[counter]['accuracy_val'])
                else:      # case we want plain performance
                    val = dfs[counter]['accuracy' + driftVal]#.mean()
                averTest = val[val.index[0]]
                row.append(averTest)      
                counter +=1
            matrix_drift.append(row)
        return matrix_drift



def performance_drift_heatmap(df, drift = 'Gaussian', params = ['mean', 'sigma'], seed_list=[1,10,100]):
    '''function to plot heatmap with drift params (mean and delta) varying and as values the accuracy of the model on drifted val'''

    filtered = df[(df['drift']==drift)&(df['seed_training'] == seed_list[0])&(df['seed_drift'] == seed_list[1])&(df['seed_metrics'] == seed_list[2])]           # filter on my dimensionality reduction and drift type          # filter on my dimensionality reduction and drift type
    gkk = filtered.groupby(params)       # group for my params (mean, delta), (a,b)
    dfs = [x[1] for x in list(gkk)]         # I have list, each element is a dataframe with fixed mean and delta. The order is: fixed mean and delta growing
    
    gmean =  filtered.groupby(params[0])      #to have all values of my parameters
    gdelta =  filtered.groupby(params[1])
    meanId = list(gmean.groups.keys())
    deltaId = list(gdelta.groups.keys())
 

    def singlePlot(meanId, deltaId, dfs, DRIFT, ax, DIFF = False):
        '''Function to have the single plots of the figure: the first heatmap withou drift, the second heatmap with drift'''
        if(DRIFT):
            title = f'accuracy on {drift} drifted Target set'
        else:
            title = 'accuracy on non-drifted Target set'
        matrix_val = get_matrix_drift(meanId,deltaId, dfs, DRIFT= DRIFT, DIFF=DIFF)
        df_val = pd.DataFrame(data = matrix_val, index = meanId, columns = deltaId)    #create dataframe
        sn.heatmap(data = df_val, linewidths=2, annot=True, vmin=0.1, vmax=1, cmap="Greens", ax=ax)                                        # plot it

       

        ax.set_title(title)
        ax.set_xlabel(labels[params[1]])
        ax.set_ylabel(labels[params[0]])

    ngraphs = 1
    # ngraphs = 2
    fig, axes = plt.subplots (ngraphs, 2, figsize = (24, 6*ngraphs))
    singlePlot(meanId, deltaId, dfs, DRIFT = False, ax = axes[0])
    singlePlot(meanId, deltaId, dfs, DRIFT = True, ax = axes[1])



# DETECTION ACCURACY

def testTOdetect(test_name, data, levels):
    ''' function that takes data and names of tests we want (test column's names, could be drift or val) and levels. It returns a list where each
      element is a vector of boolean. It represent the detected samples from the test. We assume that data is already filtered for a particular type of 
    drift and dim_red'''
    #!!! data si empty
    # DRIFT
    detected_list = []
    for counterTest in range (0, int((len(test_name)-1)/2)):
        selected = test_name[counterTest*2:counterTest*2+2] # is ttest_drift_0, ttest_drift_1
        t_test = data[selected]
        
        if(counterTest==2): 
            bonfP = (t_test > [levels[counterTest], levels[counterTest]])      # case of KL div 
            if (t_test.iloc[0,0] == 0.0):                                       # if first component is strange
                bonfP.iloc[:,0] = np.ones(len(t_test), dtype=bool)
            if (t_test.iloc[0,1] == 0.0):                                       # if first component is strange
                bonfP.iloc[:,1] = np.ones(len(t_test), dtype=bool)
        else: 
            bonfP = (t_test < [levels[counterTest], levels[counterTest]])       # case of t-test and KS-test
            
            if ((np.isnan(t_test.iloc[0,0]) or (t_test.iloc[0,0] == 1.0))):     # case of nan for t-test or 1 for ks-test for first component or (0.0) for KL_div
                bonfP.iloc[:,0] = np.ones(len(t_test), dtype=bool)
                print('--------------- EXCEPTION! FIRST COMPONENT----------------')
            # second component
            if ((np.isnan(t_test.iloc[0,1]) or (t_test.iloc[0,1] == 1.0))):     # case of nan for t-test or 1 for ks-test for first component or (0.0) for KL_div
                bonfP.iloc[:,1] = np.ones(len(t_test), dtype=bool)
                print('--------------- EXCEPTION! SECOND COMPONENT----------------')

                

        detect = np.logical_and(bonfP.iloc[:,0], bonfP.iloc[:,1])   # Bonferroni method 
        detected_list.append(detect) 

    MMD_test = data[test_name[-1]]          # case of MMD_test
    detect = MMD_test > levels[-1]
    detected_list.append(detect) 
    return detected_list

def detect(df, dim_red, thresholds, drift , seed_list):
    '''see which samples have detected the drift with respect to a particular drift and dimensionality reduction.
    To do this it calls testTOdetect'''
    filtered = df[(df['drift'] == drift)&(df['dim_red'] == dim_red)&(df['seed_training'] == seed_list[0])&(df['seed_drift'] == seed_list[1])&(df['seed_metrics'] == seed_list[2])]           # filter on my dimensionality reduction and drift type
    test_drift = ['ttest_drift_0','ttest_drift_1','ks_drift_0','ks_drift_1','kl_drift_0','kl_drift_1','MMD_drift']  #columns of drifted tests
    test_val = ['ttest_val_0','ttest_val_1','ks_val_0','ks_val_1','kl_val_0','kl_val_1','MMD_val']      # columns of not drifted tests
   
    levels = thresholds[dim_red]        #from dictionary thresholds I get levels as the list of thresholds for tests after key dim_red
 
    detected_drift = testTOdetect(test_drift, filtered, levels)         # detected_drift is list of vectors with bool of whether the test detected that drift or not
    detected_val = testTOdetect(test_val, filtered, levels)             # detected_val is as detected_drift but with tests on plain validation.

    zipped = list(zip(detected_drift[0], detected_drift[1], detected_drift[2], detected_drift[3]))
    dfDrifted = pd.DataFrame(zipped, columns=['t_test', 'ks_test', 'kl_div', 'MMD'])            # create dataframe for drifted tests

    zipped = list(zip(detected_val[0], detected_val[1], detected_val[2], detected_val[3]))
    dfVal = pd.DataFrame(zipped, columns=['t_test', 'ks_test', 'kl_div', 'MMD'])                # create dataframe for tests on plain validation

    return dfDrifted, dfVal

def detect_acc(df, thresholds, drift = 'Gaussian', seed_list=[1, 10, 100]):
    dfDrifted_list =[]                                          # it will be list with drifted tests, one for each of the dim_red
    dfVal_list = []                                             # it will be list with tests on plain validation
    for dim_red in list(thresholds.keys()):                     #for each dimensionality reduction I create dataframe for drift and for val
        dfDrifted, dfVal = detect(df, dim_red, thresholds, drift=drift, seed_list=seed_list)  
        dfDrifted_list.append(dfDrifted)
        dfVal_list.append(dfVal)
    return dfDrifted_list, dfVal_list


def print_detection_accuracy(df, thresholds, drift, seed_list):
    dfDrifted_list, dfVal_list = detect_acc(df, thresholds, drift, seed_list)
    dimRed_list = list(thresholds.keys())
    with open ('detectionAccuracy.txt', 'w') as file:

        for i in range(len(dimRed_list)):      # loop on the dim_red
            Newdf = dfDrifted_list[i]      # dataFrame for selected dim_red
            Newdf_val = dfVal_list[i]
            dim_red = dimRed_list[i]    # name of dim_red
            for col in Newdf:              # each column is different type of test
                detectAcc = sum(Newdf[col])/len(Newdf)
                file.write(f'detection accuracy on DRIFTED  with {dim_red} and test {col} : {detectAcc} \n ')
                detectAcc_val = sum(Newdf_val[col])/len(Newdf_val)
                file.write(f'detection accuracy on VAL with {dim_red} and test {col} : {detectAcc_val} \n \n')




