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


def split_indices(data, train_size, val_size):
    train_idx = int(train_size * data.shape[0])
    val_idx = int((train_size + val_size) * data.shape[0])
    return train_idx, val_idx

def make_stationary(data):
    diffed = data.copy()
    diffed = data.diff()
    return diffed

def inverse_stationary(original, yhat):
    return yhat.to_numpy() + original.loc[:-1].to_numpy()

def scale(data, avail_idx, scaler):
    scaler.fit(data[:avail_idx])
    return scaler.transform(data)

def unscale(data, scaler):
    return scaler.inverse_transform(data)

def sequence(data, seq_len, batch_sz, shuffle=True):
    x = data[:-seq_len].to_numpy(dtype=np.float32).reshape((-1, 1))
    y = data[seq_len:].to_numpy(dtype=np.float32).reshape((-1, 1))
    ds = timeseries_dataset_from_array(
        x, 
        y, 
        sequence_length=seq_len,
        batch_size=batch_sz,
        shuffle=shuffle
    )
    return ds


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

  def fit_transform(self, x):
    self.mu = np.mean(x, axis=(0), keepdims=True)
    self.sd = np.std(x, axis=(0), keepdims=True)
    normalized_x = (x - self.mu)/self.sd
    return normalized_x
    
  def transform(self, x):
    normed_x = (x - self.mu)/self.sd
    return normed_x

  def inverse_transform(self, x):
    return (x*self.sd) + self.mu