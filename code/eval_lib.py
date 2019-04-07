#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import random
import sys

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

import mapping_lib
import gamma_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)


def fetch_training_data(datadir):
  data = mapping_lib.get_mapping('variant', 'relgamma', datadir, dose='sober')
  data = training_lib.filter_for_training(data, datadir)
  data = data.dropna()
  data.reset_index(inplace=True)
  var_orig = mapping_lib.get_mapping('variant', 'original', datadir).original
  var_pam = mapping_lib.get_mapping('variant', 'pam', datadir).pam
  data['original'] = data.variant.map(var_orig)
  data['pam'] = data.variant.map(var_pam)
  data.set_index('variant', inplace=True)
  return data

def featurize_training_data(encoder, data, datadir):
  data = data.reset_index()
  encodings = data.apply(encoder, axis=1)
  Xframe = encodings.set_index(data.index)
  Xframe = training_lib.expand_dummies(Xframe)
  X = np.array(Xframe, dtype=float)
  y = np.array(data[['relgamma']], dtype=float)
  return X, y

def scale_training_data_linear(X, y):
  """Build X/y scalers

  Returns (X, X_scaler, y, y_scaler):
    X_scaler: scaler fit to full original dataset feature-space
    y_scaler: scaler fit to full original dataset outputs
    X: X_xcaler.transform(X)
    y: y_scaler.transform(y)
  """
  X_scaler = StandardScaler()
  X = X_scaler.fit_transform(X)
  y_scaler = StandardScaler()
  y = y_scaler.fit_transform(y)
  return (X_scaler, y_scaler, X, y)

def plot_confusion(data, plotdir):
  data = data.dropna()
  bins = gamma_lib.relgamma_bins()
  predbins = gamma_lib.bin_relgammas(data.y_pred, bins)
  measbins = gamma_lib.bin_relgammas(data.y_meas, bins)
  cm = confusion_matrix(measbins, predbins)
  truenorm_cm = normalize(cm, axis=1, norm='l1')

  plt.figure(figsize=(8,6))
  heatmap = sns.heatmap(truenorm_cm, annot=True, fmt=".2f", cmap='YlGnBu',
                        xticklabels=True,
                        yticklabels=True)
  # heatmap.yaxis.set_ticklabels(classes)#, rotation=0, ha='right')
  # heatmap.xaxis.set_ticklabels(classes)#, rotation=45, ha='right')
  plt.title('Gamma Confusion Matrix\n[Norm: Measured]')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plotfile = plotdir / 'confusion.truenorm.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

  prednorm_cm = normalize(cm, axis=0, norm='l1')
  plt.figure(figsize=(8,6))
  heatmap = sns.heatmap(prednorm_cm, annot=True, fmt=".2f", cmap='YlGnBu',
                        vmin=0.0, vmax=1.0,
                        xticklabels=True, yticklabels=True)
  # heatmap.yaxis.set_ticklabels(classes)#, rotation=0, ha='right')
  # heatmap.xaxis.set_ticklabels(classes)#, rotation=45, ha='right')
  plt.title('Gamma Confusion Matrix\n[Norm: Predicted]')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plotfile = plotdir / 'confusion.prednorm.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()
