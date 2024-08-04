from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras_tuner
from dataclasses import dataclass
from typing import List
from keras import layers, ops
from keras.utils import timeseries_dataset_from_array
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


#def get_model(batch_size, optimizer, loss, metrics, hp, seperate_output_layer=True):
#    model = keras.Sequential()    
#    model.add(keras.Input(
#        shape=(20, 1),
#        batch_size=batch_size,
#        name="Inputs"
#    ))
#    for idx, l in enumerate(hp.lstm_layers):
#        model.add(layers.LSTM(
#            l.units,
#            activation=l.activation,
#            recurrent_activation=l.recurrent_activation,
#            kernel_initializer=l.kernel_initializer,
#            recurrent_initializer=l.recurrent_initializer,
#            bias_initializer=l.bias_initializer,
#            dropout=l.dropout,
#            recurrent_dropout=l.recurrent_dropout,
#            return_sequences=l.return_sequences,
#            return_state=l.return_state,
#            stateful=l.stateful,
#        )
#      )
#    if seperate_output_layer:
#        model.add(layers.Dense(1))
#    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#    return model


def build_model(hp):
  model = keras.Sequential()    
  model.add(keras.Input(
      shape=(20, 1),
      batch_size=64,
      name="Inputs"
  ))  
#  units_0 = 32 #hp.Int(f"units_0", min_value=32, max_value=96, step=32)
  if hp.Boolean("selu"):
    model.add(layers.Dense(32, activation="selu", kernel_initializer="lecun_normal"))
  else:
    model.add(layers.Dense(32, activation=hp.Choice(f"activation", ["relu", "silu", "gelu"])))
  num_layers = 2 #hp.Int("num_layers", 1, 3)
  for i in range(num_layers):
#    if i == 0:
#      u = units_0
#    else:
#      u = hp.Int(f"units_{i}", min_value=32, max_value=96, step=32)
    activations = ["sigmoid", "tanh"]
    inits = ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]
    model.add(
      layers.LSTM(
        units=32, #u,
        activation="tanh", #hp.Choice(f"activation_{i}", activations),
        recurrent_activation="sigmoid",#hp.Choice(f"recurrent_activation_{i}", activations),
        dropout=hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.5),
        recurrent_dropout=hp.Float(f"recurrent_dropout_{i}", min_value=0.0, max_value=0.2),
        return_sequences=(i != num_layers - 1),
        kernel_initializer="glorot_uniform", #hp.Choice(f"kernel_initializer_{i}", inits),
        recurrent_initializer="orthogonal", #hp.Choice(f"recurrent_initializer_{i}", inits),
        bias_initializer="zeros"#""hp.Choice(f"bias_initializer_{i}", inits+["zeros"])
      ))
  model.add(
    layers.Dense(
      1,
      activation="linear", #"tanh",#hp.Choice(f"activation_{num_layers}", activations)
      kernel_initializer=hp.Choice(f"kernel_initializer_{num_layers}", inits),
      bias_initializer="zeros" #hp.Choice(f"bias_initializer_{num_layers}", inits+["zeros"]),
    ))
  learning_rate=hp.Float("lr", min_value=5e-5, max_value=4e-3, sampling="log")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate), #, clipnorm=1.0),
      loss="mse",
      metrics=["mse"],
  )
  return model


def build_spec_model(
  num_layers, 
  units, 
  activations, 
  recurrent_activations, 
  dropouts, 
  recurrent_dropouts, 
  learning_rate
):
  model = keras.Sequential()    
  model.add(keras.Input(
      shape=(20, 1),
      batch_size=64,
      name="Inputs"
  ))
  model.add(layers.Dense(32, activation="relu"))
  for i in range(num_layers):
    model.add(
      layers.LSTM(
        units=units[i],
        activation=activations[i],
        recurrent_activation=recurrent_activations[i],
        dropout=dropouts[i],
        recurrent_dropout=recurrent_dropouts[i],
        return_sequences=(i != num_layers - 1)
      ))
  model.add(
    layers.Dense(
      1,
      activation="linear"#"sigmoid"#activation=hp.Choice(f"activation_{num_layers}", ["sigmoid", "relu"])
    ))
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),#, clipnorm=1.0),
      loss="mse",
      metrics=["mse"],
  )
  return model


def get_tuner(max_trials, execs_per_trial, project_name=None):
  if project_name == None:
    project_name = "stock_predictor_" + str(datetime.now())
  tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=max_trials,
    executions_per_trial=execs_per_trial,
    overwrite=True,
    directory="tuning",
    project_name=project_name,
  )
  return tuner


