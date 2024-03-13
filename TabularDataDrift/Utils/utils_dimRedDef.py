import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import tensorflow as tf

import umap.umap_ as umap

from sklearn.decomposition import PCA

def scale_single_dataset(X):
    standard_scaler = MinMaxScaler()
    x_scaled = pd.DataFrame(
        standard_scaler.fit_transform(X),
        columns=X.columns
    )
    return x_scaled, standard_scaler
    

def scale_datasets(x_train, x_test, x_val):
  """
  Standard Scale test and train data
  """
  standard_scaler = MinMaxScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )

  x_val_scaled = pd.DataFrame(
      standard_scaler.transform(x_val),
      columns = x_val.columns
  )
  return x_train_scaled, x_test_scaled,  x_val_scaled, standard_scaler
  


class AutoEncoders(Model):      #sublass of Model class in tensorflow           #prova orthogonal, prova con torch #togli relu ultimo layer togli sigmoid alla fine

  def __init__(self, output_units, seed):
    
    super().__init__()

    self.encoder = Sequential(
        [
          Dense(32, activation="relu", kernel_initializer=initializers.GlorotNormal(seed=seed), name = 'mySequential'),
          Dense(16, activation="relu"),
          Dense(2)   # we end with only two features
        ]
    )
    
    self.decoder = Sequential(
        [
          Dense(16, activation="relu"),
          Dense(32, activation="relu"),
          Dense(output_units)
        ]
    )


  def call(self, inputs):

    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
  


def U_autoencoder_init(seed, X_train_scaled):
    '''Initialize untrained autoencoder, return the encoder'''
    auto_encoder = AutoEncoders(len(X_train_scaled.columns), seed)
    auto_encoder.compile(
        loss='mae',         # mean absolute error 
        metrics=['mae'],
        optimizer='adam'    # adam optimization 
    )
    encoder_layer = auto_encoder.encoder

    return encoder_layer



def T_autoencoder_init(seed, x_train_scaled, x_test_scaled):
    '''Initialize and train autoencoder, return encoder'''
    T_auto_encoder = AutoEncoders(len(x_train_scaled.columns), seed)
    T_auto_encoder.compile(
        loss='mae',         # mean absolute error 
        metrics=['mae'],
        optimizer='adam'    # adam optimization 
    )
    history = T_auto_encoder.fit(x_train_scaled, x_train_scaled, 
                epochs=15, 
                shuffle=True,
                validation_data=(x_test_scaled, x_test_scaled)
            )
    T_encoder_layer = T_auto_encoder.encoder 
    return T_encoder_layer

def reducer_umap_fit (seed, X_test):
    '''initialize UMAP reducer'''
    reducer = umap.UMAP(random_state=seed)
    reducer.fit(X_test)
    return reducer

def reducer_PCA_fit(X_test):
    '''initialize PCA reducer'''
    pca = PCA(n_components = 2)
    pca.fit(X_test)
    return pca


def initialize_DimReduction(seed_metrics, X_test,  X_train_scaled, X_test_scaled):
    '''Initialize all the dimensionality reduction techniques'''
    tf.random.set_seed(seed_metrics)
    # for PCA
    reducer_pca = reducer_PCA_fit(X_test)

    #for umap
    reducer_umap = reducer_umap_fit(seed_metrics, X_test)

    # for U_autoencoder
    U_encoder_layer = U_autoencoder_init(seed_metrics, X_train_scaled)

    # for T_autoencoder 
    T_encoder_layer = T_autoencoder_init(seed_metrics, X_train_scaled, X_test_scaled)

    return reducer_pca, reducer_umap, U_encoder_layer, T_encoder_layer