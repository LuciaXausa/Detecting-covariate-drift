import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import tensorflow as tf
import umap.umap_ as umap
from sklearn.decomposition import PCA


def find_dimensions_number(test, var_percentage = 0.8):
    ''' Function to find number of dimensions to reduce the dataset to
    test: dataframe on which apply PCA
    var_percentage: variance percentage we want to preserve with PCA on k dimensions
    k: int that represents the number of dimensions we will reduce before applying tests.'''
    pca = PCA()
    pca.fit(test) # a df is needed to fit the PCA reducer
    variance_ratio = pca.explained_variance_ratio_ # its length is the number of default components in the PCA; its content is the variance ration for each component returned by the PCA
    sum = 0
    k =0

    while sum < var_percentage:
        sum += variance_ratio[k]
        k += 1
    
    return k



def scale_dataset(x, standard_scaler):
  """
  Given dataframe x and standard_scaler, scale X with the defined scaler
  x: dataframe, dataset to scale
  standard_scaler: min max scaler already initialized
  """
  x_scaled = pd.DataFrame(
      standard_scaler.transform(x),
      columns=x.columns
  )

  return x_scaled
  

def init_scaler(X):
    '''Initialize the standard scaler as a min_max scaler fitted on the dataframe X in input
    X: dataframe to fit MinMaxScaler'''
    standard_scaler = MinMaxScaler()
    standard_scaler.fit(X) 

    return standard_scaler
    

class AutoEncoders(Model):      #sublass of Model class in tensorflow           

  def __init__(self, output_units, seed, k):        # k number of dimensionalities we want to reduce to
        
    super().__init__()

    self.encoder = Sequential(
        [
          Dense(16*k, activation="relu", kernel_initializer=initializers.GlorotNormal(seed=seed), name = 'mySequential'),
          Dense(8*k, activation="relu"),
          Dense(k)   # we end with only k features
        ]
    )
    
    self.decoder = Sequential(
        [
          Dense(8*k, activation="relu"),
          Dense(16*k, activation="relu"),
          Dense(output_units)
        ]
    )


  def call(self, inputs):

    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
  


def U_autoencoder_init(seed, X_train_scaled, k):
    '''Initializes untrained autoencoder, returns the encoder
    seed: seed for initialize autoencoder for reproducibility
    X_train_scaled: scaled train dataframe to initialize the autoencoder on.
    k: number of dimensions we want to reduce our dataframe to. Therefore it is the number of dimensions of the output of the last encoder's layer.'''
    auto_encoder = AutoEncoders(len(X_train_scaled.columns), seed, k)
    auto_encoder.compile(
        loss='mae',         # mean absolute error 
        metrics=['mae'],
        optimizer='adam'    # adam optimization 
    )
    encoder_layer = auto_encoder.encoder

    return encoder_layer



def T_autoencoder_init(seed, x_train_scaled, x_test_scaled, k):
    '''Initialize and train autoencoder, return encoder.
    seed: seed for initialize autoencoder for reproducibility
    X_train_scaled: scaled train dataframe to initialize and train the autoencoder on.
    X_test_scaled: scaled test dataframe to evaluate the autoencoder on.
    k: number of dimensions we want to reduce our dataframe to. Therefore it is the number of dimensions of the output of the last encoder's layer.'''
    T_auto_encoder = AutoEncoders(len(x_train_scaled.columns), seed, k)
    T_auto_encoder.compile(
        loss='mae',         # mean absolute error 
        metrics=['mae'],
        optimizer='adam'    # adam optimization 
    )
    T_auto_encoder.fit(x_train_scaled, x_train_scaled, 
            epochs=15, 
            shuffle=True,
            validation_data=(x_test_scaled, x_test_scaled)
        )
    T_encoder_layer = T_auto_encoder.encoder 
    return T_encoder_layer

def reducer_umap_fit (seed, X_test, k):
    '''initialize UMAP reducer with k components to reduce the dataframes to
    seed: int seed, used to set the umap's random state
    X_test: dataframe to fit UMAP in initalization
    k: number of components we want the umap to reduce dataframes to.
    '''
    reducer = umap.UMAP(n_components=k, random_state=seed)
    reducer.fit(X_test)
    return reducer

def reducer_PCA_fit(X_test, k):
    '''initialize PCA reducer to reduce to k components
    X_test: dataframe to fit PCA in initalization
    k: number of components we want the PCAto reduce dataframes to.
    '''
    pca = PCA(n_components = k)
    pca.fit(X_test)
    return pca


def initialize_DimReduction(seed_metrics, X_test,  X_train_scaled, X_test_scaled, k=2):
    '''Initialize all the dimensionality reduction techniques to reduce dimensions to k dimensions
    seed_metrics: int value, to set seed for initialization of dimensionality reduction techniques
    X_test: dataset to fit the dimensionality reductors in their initialization
    X_train_scaled: scaled dataframe, used for initialization of autoencoders
    X_test_scaled: scaled dataframe, used for training of autoencoder as validation data
    k: number of dimensions we want our reductors to reduce dataframes to.'''
    tf.random.set_seed(seed_metrics)
    # for PCA
    reducer_pca = reducer_PCA_fit(X_test, k)
    #for umap 
    reducer_umap = reducer_umap_fit(seed_metrics, X_test, k)
    # for U_autoencoder
    U_encoder_layer = U_autoencoder_init(seed_metrics, X_train_scaled, k)
    # for T_autoencoder 
    T_encoder_layer = T_autoencoder_init(seed_metrics, X_train_scaled, X_test_scaled, k)

    return reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer

