import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from dataclasses import dataclass
from typing import List
from keras import layers, ops
from keras.utils import timeseries_dataset_from_array
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass
class LSTMLayerHyperparams:
    units: int
    activation: str = "relu"
    recurrent_activation: str = "relu"
    kernel_initializer: str = "glorot_uniform"
    recurrent_initializer: str = "orthogonal"
    bias_initializer: str = "zeros"
    dropout: float = 0.0
    recurrent_dropout: float = 0.0
    return_sequences: bool = False
    return_state: bool = False
    stateful: bool = False

@dataclass
class ModelHyperparams:
    lstm_layers: List[LSTMLayerHyperparams]

