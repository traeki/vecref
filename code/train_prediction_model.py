#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
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
_K_FOLD_SPLITS = 4
_REL_PLOT_MIN = -1.2
_REL_PLOT_MAX = 0.5


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

cross_predictions = np.zeros_like(y)

plotdir = (UNGD / _CODEFILE).with_suffix('.plots')
shutil.rmtree(plotdir, ignore_errors=True)
plotdir.mkdir(parents=True, exist_ok=True)
modeldir = (UNGD / _CODEFILE).with_suffix('.models')
shutil.rmtree(modeldir, ignore_errors=True)
modeldir.mkdir(parents=True, exist_ok=True)
# Loop cross-validation
for i, (train, test) in enumerate(kfolder):
  # Create model
  model = training_lib.build_conv_net_model()
  # Feed training Data
  model_history = model.fit(X[train], y[train],
                            batch_size=32, epochs=30,
                            validation_data=(X[test], y[test]))
  modelfile = modeldir / 'model.{i}.d5'.format(**locals())
  modelfile = str(modelfile)  # NOTE(jsh): workaround until Keras PR #11466
  model.save(modelfile)
  coverfile = modeldir / 'model.{i}.coverage.pickle'.format(**locals())
  pickle.dump(data.index[test], open(coverfile, 'wb'))

  cross_predictions[test] = model.predict(X[test]).ravel()

  plt.figure(figsize=(6,6))
  plt.plot(model_history.history['mean_squared_error'])
  plt.plot(model_history.history['val_mean_squared_error'])
  plt.title('Model Error [Fold {i}]'.format(**locals()))
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.ylim(-0.2, 1.2)
  plt.legend(['Train', 'Test'], loc='upper right')
  plotfile = plotdir / 'error.trace.{i}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

  plt.figure(figsize=(6,6))
  plt.scatter(cross_predictions[test], y[test], marker='.', alpha=.2)
  plt.title('Model Predictions [Fold {i}]'.format(**locals()))
  plt.xlabel('predicted')
  plt.ylabel('measured')
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plotfile = plotdir / 'scatter.{i}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

plt.figure(figsize=(6,6))
plt.scatter(cross_predictions, y, marker='.', alpha=.2)
plt.title('Model Predictions [agg]'.format(**locals()))
plt.xlabel('Predicted relative γ')
plt.ylabel('Measured relative γ')
plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plotfile = plotdir / 'scatter.agg.png'.format(**locals())
plt.savefig(plotfile)
plt.close()

locusmap = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
genemap = mapping_lib.get_mapping('locus_tag', 'gene_name', UNGD)
geneids = genemap.loc[locusmap.loc[data.index].locus_tag]
geneids.index = data.index
data['y_meas'] = data.relgamma
data['y_pred'] = cross_predictions
data['gene_name'] = geneids
data['original'] = familymap.original

mapping_lib.make_mapping(data.reset_index(), 'variant', 'y_pred', UNGD)

for gene, group in data.groupby('gene_name'):
  predicted = group.y_pred
  measured = group.y_meas
  sprrho, _ = st.spearmanr(predicted, measured)
  prsrho, _ = st.pearsonr(predicted, measured)
  plt.figure(figsize=(6,6))
  template = 'Predictions vs. Measurements\n{gene}'
  main_title_str = template.format(**locals())
  plt.title(main_title_str)
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.xlabel('Predicted relative γ')
  plt.ylabel('Measured relative γ')
  template = 'Spearman: {sprrho:.2f}\nPearson: {prsrho:.2f}'
  subgrouper = group.groupby('original')
  for original, subgroup in subgrouper:
    predicted = subgroup.y_pred
    measured = subgroup.y_meas
    g = plt.scatter(predicted, measured, s=6, alpha=1)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plt.text(_REL_PLOT_MAX - 0.1, _REL_PLOT_MIN + 0.1, template.format(**vars()))
  plt.tight_layout()
  plotfile = plotdir / 'scatter.agg.{gene}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

# import IPython; IPython.embed()

# TODO(jsh): Change loss function or input data to
# TODO(jsh):  -- heavily emphasize interior values
# TODO(jsh):  -- bound values to intended range
# TODO(jsh):  -- de-emphasize exreme values
# TODO(jsh):  -- sample based on distribution
# TODO(jsh):  -- all of the above
# TODO(jsh):  -- ...?


# TODO(jsh): Tune hyper-parameters
# TODO(jsh): Does gamma format (+/-1, etc.) matter?
# TODO(jsh): Consider resampling/imbalanced feed
