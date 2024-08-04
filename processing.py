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


class Stationarizer:
  def __init__(self):
    self.orig_data = None

  def fit_transform(self, data):
    self.orig_data = data.copy()
    return np.array([data[i] - self.orig_data[i-1] for i in range(1, len(data))])

  def inverse_transform(self, data):
    return np.array([data[i] + self.orig_data[i] for i in range(len(data))])


class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def transform(self, x):
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu
