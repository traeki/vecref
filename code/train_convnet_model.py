#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import preprocessing as skpreproc
from sklearn import model_selection as skmodsel

import gamma_lib
import mapping_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name
PLOTDIR = (UNGD / _CODEFILE).with_suffix('.plots')

_DATA_FRACTION = 1
_K_FOLD_SPLITS = 3
_REL_PLOT_MIN = -1.2
_REL_PLOT_MAX = 1
_BATCH_SIZE = 32
_EPOCHS = 10


######################
# Read Relgamma Data #
######################
data = mapping_lib.get_mapping('variant', 'relgamma', UNGD)

###############
# Filter Data #
###############
# Remove non-oneoff guides (parents, off-strand, controls, etc.)
data = training_lib.filter_for_training(data, UNGD)
data = data.dropna()

###############
# Weight Data #
###############
# binmap = mapping_lib.get_mapping('variant', 'rgbin', UNGD).loc[data.index]
# binweights = gamma_lib.weight_bins(binmap.rgbin)
# weightmap = binmap.rgbin.map(binweights)
# weightmap.name = 'binweight'
# mapping_lib.make_mapping(weightmap.reset_index(), 'variant', 'binweight', UNGD)

###################
# Downsample Data #
###################
data = training_lib.downsample_families(data, _DATA_FRACTION, UNGD)

###################
# Preprocess Data #
###################
encoder = training_lib.one_hot_pair_encoder(UNGD)
encodings = [encoder(x)[1] for x in data.index]
X = np.stack(encodings, axis=0)
y = np.array(data[['relgamma']], dtype=float)

X_scaler = dict()
for i in range(X.shape[1]):
  for j in range(X.shape[3]):
    X_scaler[(i,j)] = skpreproc.StandardScaler()
    X[:,i,:,j] = X_scaler[(i,j)].fit_transform(X[:,i,:,j])
y_scaler = skpreproc.StandardScaler()
y = y_scaler.fit_transform(y)

###################################
# Split Data for Cross-Validation #
###################################
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)
familymap = familymap.loc[data.index]
kfolder = skmodsel.GroupKFold(_K_FOLD_SPLITS).split(X, y, familymap)

# weights = weightmap[data.index]

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)
modeldir = training_lib.CONVNET_MODELDIR
shutil.rmtree(modeldir, ignore_errors=True)
modeldir.mkdir(parents=True, exist_ok=True)
# Loop cross-validation
for i, (train, test) in enumerate(kfolder):
  # Create model
  model = training_lib.build_conv_net_model()
  # Feed training Data
  model_history = model.fit(X[train], y[train],
                            batch_size=_BATCH_SIZE, epochs=_EPOCHS,
                            validation_data=(X[test], y[test]))
  modelfile = modeldir / 'model.{i}.d5'.format(**locals())
  modelfile = str(modelfile)  # NOTE(jsh): workaround until Keras PR #11466
  model.save(modelfile)
  coverfile = modeldir / 'model.{i}.coverage.pickle'.format(**locals())
  pickle.dump(data.index[test], open(coverfile, 'wb'))

  plt.figure(figsize=(6,6))
  plt.plot(model_history.history['mean_squared_error'])
  plt.plot(model_history.history['val_mean_squared_error'])
  plt.title('Model Error [Fold {i}]'.format(**locals()))
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.ylim(-0.2, 1.2)
  plt.legend(['Train', 'Test'], loc='upper right')
  plotfile = PLOTDIR / 'error.trace.{i}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()
