#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import AvgPool2D
from keras.layers import Dropout
from keras.models import Sequential

from sklearn.preprocessing import normalize

import mapping_lib
import gamma_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)


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
  plt.title('Absolute gamma Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plotfile = plotdir / 'confusion.truenorm.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

  prednorm_cm = normalize(cm, axis=0, norm='l1')
  plt.figure(figsize=(8,6))
  heatmap = sns.heatmap(prednorm_cm, annot=True, fmt=".2f", cmap='YlGnBu',
                        xticklabels=True,
                        yticklabels=True)
  # heatmap.yaxis.set_ticklabels(classes)#, rotation=0, ha='right')
  # heatmap.xaxis.set_ticklabels(classes)#, rotation=45, ha='right')
  plt.title('Absolute gamma Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plotfile = plotdir / 'confusion.prednorm.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()
