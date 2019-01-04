#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from keras.models import load_model

import gamma_lib
import mapping_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name


######################
# Read Prediction Data #
######################
predmap = mapping_lib.get_mapping('variant', 'y_pred', UNGD)
measmap = mapping_lib.get_mapping('variant', 'relgamma', UNGD)

modeldir = UNGD / 'train_prediction_model.models'
model_template = 'model.{i}.d5'
coverage_template = 'model.{i}.coverage.pickle'

# Loop over models
models = list()
coverage = list()
i = 0
while True:
  try:
    # Create model
    modelfile = modeldir / model_template.format(**locals())
    model = load_model(str(modelfile))
    models.append(model)
    coverfile = modeldir / coverage_template.format(**locals())
    cover = pickle.load(open(coverfile, 'rb'))
    coverage.append(cover)
    print('Found model {i}.'.format(**locals()))
    i += 1
  except OSError:
    break

def never():
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

# NOTE(jsh): FORCES END OF FILE
sys.exit(0)

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

mapping_lib.make_mapping(data, 'variant', 'y_pred', UNGD)

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
