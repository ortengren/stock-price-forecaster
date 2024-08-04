import json
import copy
import joblib
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression

from processing import Stationarizer, Normalizer
from plotting import visualize_loss, show_plot
from model_builder import build_model, build_spec_model, get_tuner


def sort(array):
    less = []
    equal = []
    greater = []
    if len(array) > 1:
        pivot = array[0]["score"]
        for x in array:
            if x["score"] < pivot:
                less.append(x)
            elif x["score"] == pivot:
                equal.append(x)
            elif x["score"] > pivot:
                greater.append(x)
        return sort(less)+equal+sort(greater)
    else:
        return array


def get_trials(num_trials, proj_name):
  trials = list()
  for n in range(num_trials):
    path = f"tuning/{proj_name}/trial_{n:03d}/trial.json"
    with open(path, mode="r") as f:
      trial = json.load(f)
      trials.append(trial)
  return sort(trials)


def get_RMSEs(trials):
  tmp = copy.deepcopy(trials)
  rmse_trials = list()
  for trial in tmp:
    trial.update({"rmse" : np.sqrt(trial["score"])})
    rmse_trials.append(trial)
  return rmse_trials


def make_histogram(rmse_trials, num_bins=25, width=0.1):
  bins = np.linspace(min([x["rmse"] for x in rmse_trials]), max([x["rmse"] for x in rmse_trials]), num_bins)
  counts = [[cutoff, 0] for cutoff in bins]
  t = 0
  c = 0
  while t < len(rmse_trials) and c < len(counts):
    if rmse_trials[t]["score"] <= counts[c][0]:
      counts[c][1] += 1
      t += 1
    else:
      c += 1
  scx = list()
  heights = list()
  for el in counts:
    scx.append(el[0])
    heights.append(el[1])

  plt.bar(scx, heights, edgecolor="black", width=width)
  plt.xlabel("RMSE")
  plt.ylabel("number of occurrences")
  plt.title("Score distribution")
  plt.show()
  return counts


def plot_lrs(rmse_trials, log=False, lin_reg=None):
  rmse_scores = np.array([trial["rmse"] for trial in rmse_trials])
  if log:
    lrs = np.array([np.log10(trial["hyperparameters"]["values"]["lr"]) for trial in rmse_trials])
    xlabel = "log(learning rate)"
  else:
    xlabel = "learning rate"
    lrs = np.array([trial["hyperparameters"]["values"]["lr"] for trial in rmse_trials])
  plt.scatter(lrs, rmse_scores.reshape((-1, 1)))
  plt.xlabel("learning rate")
  plt.ylabel("RMSE")
  if lin_reg:
    reg.fit(lrs.reshape((-1, 1)), rmse_scores.reshape((-1, 1)))
    x = np.linspace(min(lrs), max(lrs), 100)
    y = reg.coef_ * x + reg.intercept_
    plt.plot(x, y.reshape((-1, 1)))
  plt.show()


def get_trials_df(rmse_trials):
  df = pd.DataFrame([trial["hyperparameters"]["values"] for trial in rmse_trials])
  df["trial id"] = [trial["trial_id"] for trial in rmse_trials]
  df["rmse"] = [trial["score"] for trial in rmse_trials]
  df["mse"] = [trial["score"] ** 2 for trial in rmse_trials]
  df["best step"] = [trial["best_step"] for trial in rmse_trials]
  df["total dropout"] = df["dropout_0"] + df["dropout_1"]
  return df

