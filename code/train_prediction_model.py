#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as skpreproc
from sklearn import model_selection as skmodsel

import mapping_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name

_DATA_FRACTION = 1
_K_FOLD_SPLITS = 10


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
# Group by family, sample to 1%
data = training_lib.downsample_families(data, _DATA_FRACTION, UNGD)

###################
# Preprocess Data #
###################
encoder = training_lib.one_hot_pair_encoder(UNGD)
X = np.stack([encoder(x)[1] for x in data.index], axis=0)
y = np.array(data.relgamma)

###################################
# Split Data for Cross-Validation #
###################################
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)
familymap = familymap.loc[data.index]
kfolder = skmodsel.GroupKFold(_K_FOLD_SPLITS).split(X, y, familymap)


plotdir = (UNGD / _CODEFILE).with_suffix('.plots')
shutil.rmtree(plotdir, ignore_errors=True)
plotdir.mkdir(parents=True, exist_ok=True)
# Loop cross-validation
for i, (train, test) in enumerate(kfolder):
  # Create model
  model = training_lib.build_conv_net_model()
  # Feed training Data
  model_history = model.fit(X[train], y[train],
                            batch_size=32, epochs=30,
                            validation_data=(X[test], y[test]))
  plt.plot(model_history.history['mean_squared_error'])
  plt.plot(model_history.history['val_mean_squared_error'])
  plt.title('Model Error [Fold {i}]'.format(**locals()))
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.ylim(-0.2, 1.2)
  plt.legend(['Train', 'Test'], loc='upper right')
  plotfile = plotdir / 'error.trace.{i}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.clf()

  plt.scatter(y[test], model.predict(X[test]), marker='.', alpha=.2)
  plt.title('Model Predictions [Fold {i}]'.format(**locals()))
  plt.xlabel('measured')
  plt.ylabel('predicted')
  plt.xlim(-1.2, 0.2)
  plt.ylim(-1.2, 0.2)
  plotfile = plotdir / 'scatter.{i}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.clf()

# TODO(jsh): Change loss function or input data to
# TODO(jsh):  -- heavily emphasize interior values
# TODO(jsh):  -- de-emphasize exreme values
# TODO(jsh):  -- sample based on distribution
# TODO(jsh):  -- all of the above
# TODO(jsh):  -- ...?


# TODO(jsh): Tune hyper-parameters
# TODO(jsh): Does gamma format (+/-1, etc.) matter?
# TODO(jsh): Consider resampling/imbalanced feed
