
def detect_row(row, levels, k):
    '''Given a row of data and levels, we see which test type detect the row. 
    Detected_raw : listwith 4 bool values: each one tells if the corresponding test have detected or not for the specified row.
    row: row of a dataframe, whith info of tests result
    levels: list uof list with thresholds for each test and for each component
    k: number of components'''

    
    test_type = ['t_test_', 'ks_test_', 'kl_div_']       #univariate test
    detected_row = []       # where store detection results for the row
    
    for test_i in range(len(test_type)): # for each test
        detect_test = []        # list where store all the results for each component. 
        for n_comp in range(k):     # for each dimension
            col_name = str(test_type[test_i])+str(n_comp)           # string with col_name formed by name of the test and number of component
            value = row[col_name]                                   # value as result of test on that component
            threshold = levels[test_i][n_comp]                      # get threshold value
            if (test_type == 'kl_test'):        # case of divergence: detected if value greater than threshold
                detect_comp = (value > threshold)
            else:                               # case of test p-value: detected if value minor than threshold
                detect_comp = (value < threshold)
            detect_test.append(detect_comp)     #add True or False for this component to the list of detected for the test

        detect_all_comp = all(detect_test)          #True if all items are True, else False
        detected_row.append(detect_all_comp)
    # for MMD
    MMD_detect = (row[-1] > levels[-1])     #  is discrepancy so detected if value greater than threshold
    detected_row.append(MMD_detect)
    return detected_row



def detect_drift_kdim(df, thresholds, drift , seed_list, k=2):
    '''see in which samples metrics have detected the drift with respect to a particular drift and dimensionality reduction.
    Changing k we can access to data reduced to different dimensions
    df: dataframe from a csv file, it stores all the tests' results and info for reproduce each case.
    thresholds: dict with as kays the string of the name of the dimensionality dimension techniques. Each key is associated to a list of 4 items. Each item stores the threshold for a type of test (t-test, KS-test, KL-div, MMD), so we have a k-dimensional list for univariate tests (a threshold for each component) and a single value for MMD.
    drift: string with name of type of drift we want to compute detection accuracy on.
    seed_list: list of seeds, set for filter the dataset for a particular stochastic setting
    k: number of dimensions the images' dataframes were reduced to.
    tests_list: list of lists with 4 item, each for dimensionality reductor type. Each item is a list with detection results (bool tyoe) for each type os test.
    '''
    tests_list = []    # where storing detection results for each dimensionality reduction techniques

    filtered_drift_seed = df[(df['drift'] == drift)&(df['seed_split'] == seed_list[0])&(df['seed_drift'] == seed_list[1])&(df['seed_metrics'] == seed_list[2])]  

    for dim_red in list(thresholds.keys()):                     #for each dimensionality reduction techniques

        filtered = filtered_drift_seed[(filtered_drift_seed['dim_red'] == dim_red)]           # filter on my dimensionality reduction
        levels = thresholds[dim_red]        #from dictionary thresholds I get levels as the list of thresholds for tests after key dim_red
        # where store results of drift accuracy for each row
        ttest = []
        kstest = []
        kldiv = []
        MMD = []
        for _, row in filtered.iterrows():    # iterating on the rows of filtered dataframe and detect on each of them      
            detected_row =detect_row(row, levels, k)
            ttest.append(detected_row[0])
            kstest.append(detected_row[1]) 
            kldiv.append(detected_row[2]) 
            MMD.append(detected_row[3]) 
        
        result_for_dimRed = [ttest, kstest, kldiv, MMD]
        tests_list.append(result_for_dimRed)

    return tests_list



def print_detection_accuracy(df, thresholds, drift, val, seed_list, output_file, k):
    ''' Function that compute and print detection accuracy.
    df: dataframe with results for each type of drift, dimensionality reduction techniques and statistical test. It came by a csv file.
    thresholds: dict with as keys the string of the name of the dimensionality reduction techniques. Each key is associated to a list of 4 items. Each item stores the threshold for a type of test (t-test, KS-test, KL-div, MMD), so we have a k-dimensional list for univariate tests (a threshold for each component) and a single value for MMD.
    drift: string with name of type of drift we want to compute detection accuracy on.
    val: string with name of reference data we want also to compute the detection accuracy on. This is in order to investigate if metrics give false positive.
    seed_list: list of seeds, set for filter the dataset for a particular stochastic setting
    output_file: directory of the file we want to print the thresholds on.
    k: number of dimensions the images' dataframes were reduced to.'''

    drift_list = detect_drift_kdim(df, thresholds, drift, seed_list, k)         # on drift dataset
    val_list = detect_drift_kdim(df, thresholds, val, seed_list, k)             # on val dataset

    dimRed_list = list(thresholds.keys())       # list of dim_red techniques
    test_names = ['t_test', 'ks_test', 'kl_div', 'MMD']

    with open (output_file, 'w') as file:
        for i in range(len(dimRed_list)):      # loop on the dim_red
            val_tests = val_list[i]             # 4 lists of test results           
            drift_tests = drift_list[i]
            dim_red = dimRed_list[i]    # name of dim_red
            # print separation
            file.write ('-'*20 + ' '*4 + f'{dim_red}'+ ' '*4 +'-'*20+ '\n')

            for j in range(len(val_tests)):              # j represent a different type of test
                detectAcc_drift = sum(drift_tests[j])/len(drift_tests[j])           #for each test compute accuracy
                file.write(f'detection accuracy on {drift} data  with {dim_red} and test {test_names[j]} : {detectAcc_drift} \n')
                detectAcc_val = sum(val_tests[j])/len(val_tests[j])
                file.write(f'false postitive rate on {val} data  with {dim_red} and test {test_names[j]} : {detectAcc_val} \n \n')


def read_threshold(threshold_path):
    '''Function to read threshold. 
    threshold_path: directory of the text file'''
    # read the file and save as string in threshold
    threshold_file = open(threshold_path, 'r')
    content = threshold_file.read()     
    threshold_file.close()


    modified = content.replace(' ', '') #'clean string from empty space
    sections = modified.split('\n\n')   # list with info for each dimensionality reduction techniques
    list_sections = [x.split('\n') for x in sections]   

    threshold_dict = {}     # where storing thresholds
    name_dimRed = ['NO_dim_red', 'PCA', 'UMAP', 'U_AUTOENCODER', 'T_AUTOENCODER']       #keys
    name_tests = ['t-test', 'ks-test', 'kl_div', 'MMD']     # name of different types of tests
    for i in range(len(list_sections)):   # for each dim_red technique
        key = name_dimRed[i]        # corresponding key
        threshold_for_dimRed = list_sections[i] #list of info for selected fim_red
        threshold_dict[key] = []        # list where storing thresholds. It will be a list where each item contains the corresponding thresholds for each type of test. If the test is univariate it will be a list of k values, each one is the threshold for the test applied to the corresponding component. In case of MMD is simply a value.
        for test in name_tests:     # for each type of test
            test_index = threshold_for_dimRed.index(test)   # where we find the test name in the list. We know that the corresponding values are stored in the following item
            numeric_threshold = eval(threshold_for_dimRed[test_index+1])  # since the values are as string we convert them as numerical value 
            if(type(numeric_threshold)!='tuple'):         #case of MMD: since we have a single value from eval we get a flow and not a tuple. 
                    threshold_dict[key].append(numeric_threshold) # we append directly the value
            else:       # case of univariate tests: we get a tuple and we want to store it as a list
                threshold_dict[key].append(list(numeric_threshold))
    
    return threshold_dict