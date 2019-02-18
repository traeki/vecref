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
_EPOCHS = 30
_BATCH_SIZE = 32


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

###################
# Downsample Data #
###################
data = training_lib.downsample_families(data, _DATA_FRACTION, UNGD)

###################
# Preprocess Data #
###################
var_orig = mapping_lib.get_mapping('variant', 'original', UNGD).original
var_pam = mapping_lib.get_mapping('variant', 'pam', UNGD).pam
data.reset_index(inplace=True)
data['original'] = data.variant.map(var_orig)
data['pam'] = data.variant.map(var_pam)
encoder = training_lib.get_linear_encoder()
encodings = data.apply(encoder, axis=1)

Xframe = encodings.set_index(data.variant)
Xframe = training_lib.expand_dummies(Xframe)
X = np.array(Xframe, dtype=float)
y = np.array(data[['relgamma']], dtype=float)

y_orig = y
X_scaler = skpreproc.StandardScaler()
X = X_scaler.fit_transform(X)
y_scaler = skpreproc.StandardScaler()
y = y_scaler.fit_transform(y)

###################################
# Split Data for Cross-Validation #
###################################
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)
familymap = familymap.loc[data.variant]
kfolder = skmodsel.GroupKFold(_K_FOLD_SPLITS).split(X, y, familymap)

# weights = weightmap[data.index]

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)
modeldir = training_lib.LINEAR_MODELDIR
shutil.rmtree(modeldir, ignore_errors=True)
modeldir.mkdir(parents=True, exist_ok=True)
# Loop cross-validation
for i, (train, test) in enumerate(kfolder):
  # Create model
  model = training_lib.build_linear_model(X.shape[1])
  # Feed training Data
  model_history = model.fit(X[train], y[train],
                            epochs=_EPOCHS, batch_size=_BATCH_SIZE,
                            shuffle=True,
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

model = training_lib.build_linear_model(X.shape[1])
model_history = model.fit(X, y, epochs=_EPOCHS, batch_size=_BATCH_SIZE, shuffle=True)
modelfile = str(modeldir / 'model.max.d5')
model.save(modelfile)
plt.figure(figsize=(6,6))
plt.plot(model_history.history['mean_squared_error'])
plt.title('Model Error [Full]'.format(**locals()))
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.ylim(-0.2, 1.2)
plt.legend(['Train'], loc='upper right')
plotfile = PLOTDIR / 'error.trace.max.png'.format(**locals())
plt.savefig(plotfile)
plt.close()
