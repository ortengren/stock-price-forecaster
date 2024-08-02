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

from tuner import LSTMLayerHyperparams, ModelHyperparams


def get_model(batch_size, optimizer, loss, metrics, hp, seperate_output_layer=True):
    model = keras.Sequential()    
    model.add(keras.Input(
        shape=(20, 1),
        batch_size=batch_size,
        name="Inputs"
    ))
    for idx, l in enumerate(hp.lstm_layers):
        model.add(layers.LSTM(
            l.units,
            activation=l.activation,
            recurrent_activation=l.recurrent_activation,
            kernel_initializer=l.kernel_initializer,
            recurrent_initializer=l.recurrent_initializer,
            bias_initializer=l.bias_initializer,
            dropout=l.dropout,
            recurrent_dropout=l.recurrent_dropout,
            return_sequences=l.return_sequences,
            return_state=l.return_state,
            stateful=l.stateful,
        )
      )
    if seperate_output_layer:
        model.add(layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_model(hp):
  model = keras.Sequential()    
  model.add(keras.Input(
      shape=(20, 1),
      batch_size=8,
      name="Inputs"
  ))
  num_layers = hp.Int("num_layers", 1, 4)
  for i in range(num_layers):
    model.add(
      layers.LSTM(
        units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16),
        activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
        recurrent_activation=hp.Choice(f"recurrent_activation_{i}", ["sigmoid", "relu"]),
        dropout=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.6, step=0.1),
        recurrent_dropout=hp.Float(f"recurrent_dropout_{i}", min_value=0.0, max_value=0.6, step=0.1),
        return_sequences=(i != num_layers - 1)
      ))
  model.add(
    layers.Dense(
      1,
      activation=hp.Choice(f"activation_{num_layers}", ["sigmoid", "relu"])
    ))
  learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
      loss="mse",
      metrics=["mse"],
  )
  return model


def get_tuner():
  tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="mse",
    max_trials=100,
    executions_per_trial=1,
    overwrite=True,
    directory="tuning",
    project_name="stock_predictor",
  )
  return tuner

