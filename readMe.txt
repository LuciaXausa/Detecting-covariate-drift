Methods to simulate and detect drift on input data.

On tabularDataDrift:
    Methods to simulate and detect drift on input data for tabular data.

    Two tabular numerical dataset are considered (Heart_Failure and Iris). For each of them there is a folder.

    One file is the dataset.

    The GenerateTests notebook take the dataset, train a random forest on it for classification task, simulate gaussian and linear drift on two of its features, 
    performs techniques of dimensionality reduction and save the results on the file testResults. The techniques used are of the form dimensionality reduction + statistical tests.
    Dimensionality reduction implemented methods are: no dimensionality reduction (used as reference point), PCA, UMAP, untrained autoencoder, trained autoencoder. Each of them is used to reduce to n=2 dimensions
    Statistical tests used are: t test, ks test, KL divergence on single components, MMD on whole reduced dataset. Tests are performed between test set and drifted vaildation set. 
    Same tests are performed also on test set and validation set in order to have a reference for the no-drift case.

    testResult is the file where the results of the statistics are saved as csv file.
    extended file has data with more variety of parameters value but only a fixed seed.

    plotResult is the notebook where results are investigated. 
    The first plot is an heatmap that shows values of fixed test at fixed seeds and dimensionality reduction at varying of drift parameters.
    The second plot in a Boxplot to show the difference of tests at fixed dimensionality reduction. It returns p-value for difference between drifted and non drifted results. 
    The variability of test could be setted to differend seeds.
    The third graph hilights relation between drift parameters and tests after different type of dimensionality reduction.
    The forth graph is an heatmap to investigate the fall of accuracy of classification model realted to drift intensity.
    The last function compute detection accuracy for each combination of dimensionality reduction and statistical test.
    An instance is correctly detected if the results of the tests are above/below given thresholds.

    Results of detection accuracy  are printed in the file detectionAccuracy.txt

    thresholds is the notebook where thresholds for detection accuracy are computed. They are gotten by performing same techniques of drift detection on batches of the dataset. 

    Thresholds results are printed on threshold.txt file.


    In Utils there are the used functions. 

    In DriftSimulation there are function to simulate gaussian and linear drift.
    In DriftTest there are functions to perform dimensionality reduction and statistical tests.
    In dimRedDef there are functions to initialize dimensionality reducers.
    In generate there are the main functions to be applied to the datasets. They refers to other utils file and simulate drift and perform tests on them.
    In plottingGraph there are functions to do results analysis
    In thresholds there are functions used to compute personalized threshold through permutations on dataset
    In training there are functions to train the classification model.
    In utils_llm there are functions to apply drift detection after Large Language Model application.

On imageDataDrift
    Methods to simulate and detect drift on input data for image data.

    In drift_and_thresholds two types of drift are simulated(Gaussian and change of intensity colors) on source data.
    Both source and drifted data are then pre-processed through the application of a black mask, by resizing them and by applying a pre-trained resnet.
    In addition to this thresholds are computed for each combination of dimensionality reduction technique and statistical test. 
    They are obtained by applying metrics on permutation of batches of source data. Threshold value is defined as the 5-th percentile of such results.
    Thresholds are then printed. Thresholds are computed for metrics that reduce dataframes to both 2 and 6 dimensions.
    
    In dev_test metrics are applied to development data. In particular metrics test if source data is different from synthetic drifted data.
    In this file investigation is done about how many dimensions it would be good to reduce the dataframe before applying tests.
    As done in the paper 'Failing Loudly', the number k of reduced dimensions is defined as the minimumm number of components that with PCA preserves at least the 80% of variance of the data.
    Results of such tests are saved in devResults csv files.

    In prod_test metrics are applied to production data. In particular metrics test if source data is different from real drifted data but also from data from a non-drifted day,
    in order to have a reference.
    Results of such tests are saved in prodResults csv file.

    In result_analysis there is analysis, from the csv file with test results,about which metrics were able to detect correctly the drift and which detect erroneously a drift where in fact there is not.
    A metric is considered to detect a drift if the value it returns is extreamer than the corresponding threshold. 
    In this way detection accuracy is computed and printed.

    In the folder thresholds_and_results thresholds, test results and analysis of detection are stored. 
    In particular there are two subfolders (2dim and 6dim) which cointains the same files with the only difference that all the computations are done with reductors that reduce to 2 dimensions or 6 dimensions.
    The contained files are:
        - thresholds: text file where thresholds are stored for each combination of dimensionality reduction techniques and statistical test
        - devResults: csv file where results from metrics applied on development data are stored
        - prodResult: csv file where results from metrics applied on production data are stored
        - gaussian_accuracy: text file where detection accuracy for simulated gaussian drift  and false positive rate for val data are printed. 
        - intensity_accuracy: text file where detection accuracy for simulated intensity drift and false positive rate for val data are printed.
        - production_accuracy: text file where detection accuracy for real drifted production data and false positive rate for normal production data are printed.
    
    In the folder Utils there are all the functions used in the notebooks.
        - utils_detectAcc: it cointains functions used to compute detection accuracy and false positive rate. They are: detect_row, detect_drift_kdim, print_detection_accuracy, read_threshold
        - utils_dimRedDef: it cointains functions used to define and initialize dimensionality reduction techniques. They are: find_dimensions_number, scale_dataset, init_scaler, AutoEncoders, 
            U_autoencoder_init, T_autoencoder_init, reducer_umap_fit, reducer_PCA_fit, initialize_DimReduction
        - utils_driftSimulating: it contains functions used to create synthetic drifted data and to pre-process images. They are: from_npg_to_jpg, black_filter, add_Gaussian_noise, change_intensity_greys, 
            only_image_folder, create_black_folder, create_gaussian_folder, create_intensity_folder
        - utils_DriftTest: it cointains functions used to apply statistical tests on dataframes and to print their results in csv file. They are: range01, KL, mmd_rbf, univariate_tests, statistical_tests_exe_kdim, access_components_kdim,
            get_components_kdim, print_statistical_results_kdim
        - utils_generateTests: it cointains functions used to apply metrics on data. They are: split_data, reduced_on_drift_kdim, test_on_reduced_kdim
        - utils_resNet: it cointains functions used to initialize the resnet model and get images as a dataframe. They are: init_resnet, img_transformation, feature_selection_from_img, df_from_folder
        - utils_thresholds: it cointains functions used to compute thresholds. They are: get_batches, permutation_list, test_on_permutation_kdim, permute_and_test, print_thresholds, thresholds_None, thresholds_PCA_images
            thresholds_UMAP_images, split_dataset, thresholds_AUTOENCODER_U_images, thresholds_AUTOENCODER_T_images



