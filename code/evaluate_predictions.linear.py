#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import pickle
import shutil
import sys

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import preprocessing as skpreproc

import eval_lib
import gamma_lib
import mapping_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name
PLOTDIR = (UNGD / _CODEFILE).with_suffix('.plots')

_REL_PLOT_MIN = -1.2
_REL_PLOT_MAX = 1

_FIGDPI = 300

############################
# Re-load/process raw data #
############################
data = eval_lib.fetch_training_data(UNGD)
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)
familymap = familymap.loc[data.index]
(_, X, y) = eval_lib.featurize_training_data(data, UNGD)
cross_predictions = np.full_like(y, np.nan)
y_orig = y
X_scaler, y_scaler, X, y = eval_lib.scale_training_data_linear(X, y)

########################
# Read Prediction Data #
########################
modeldir = training_lib.LINEAR_MODELDIR
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

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)
# Loop cross-validation
for i in range(len(models)):
  model = models[i]
  cover = coverage[i]
  testmask = data.index.isin(cover)
  train = (~testmask).nonzero()[0]
  test = testmask.nonzero()[0]
  train_predictions = model.predict(X[train])
  train_predictions = y_scaler.inverse_transform(train_predictions)
  train_predictions = train_predictions.reshape(-1,1)
  test_predictions = model.predict(X[test])
  test_predictions = y_scaler.inverse_transform(test_predictions)
  test_predictions = test_predictions.reshape(-1,1)
  cross_predictions[test] = test_predictions

  plt.figure(figsize=(6,6))
  plt.scatter(train_predictions, y[train], marker='.', alpha=.2, label='train')
  plt.scatter(test_predictions, y[test], marker='.', alpha=.2, label='test')
  plt.title('Predictions vs. Measurements\n[Fold {i}]'.format(**locals()))
  plt.xlabel('predicted')
  plt.ylabel('measured')
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  template = 'Train: {trainrho:.2f}\nTest: {testrho:.2f}'
  trainrho, _ = st.pearsonr(train_predictions.ravel(), y[train].ravel())
  testrho, _ = st.pearsonr(test_predictions.ravel(), y[test].ravel())
  plt.text(_REL_PLOT_MAX - 0.2, _REL_PLOT_MIN + 0.2, template.format(**locals()))
  plt.legend(loc='lower left', fontsize='small')
  plt.tight_layout()
  plotfile = PLOTDIR / 'scatter.{i}.png'.format(**locals())
  plt.savefig(plotfile, dpi=_FIGDPI)
  plt.close()

plt.figure(figsize=(6,6))
plt.scatter(cross_predictions, y, marker='.', alpha=.2, label='X-prediction')
plt.title('Predictions vs. Measurements\n[agg]'.format(**locals()))
plt.xlabel('Predicted relative γ')
plt.ylabel('Measured relative γ')
plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend(loc='lower left', fontsize='small')
plt.tight_layout()
plotfile = PLOTDIR / 'scatter.agg.png'.format(**locals())
plt.savefig(plotfile, dpi=_FIGDPI)
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

rhos = list()

for gene, group in data.groupby('gene_name'):
  predicted = group.y_pred
  measured = group.y_meas
  sprrho, _ = st.spearmanr(predicted, measured)
  prsrho, _ = st.pearsonr(predicted, measured)
  rhos.append(pd.Series({'gene':gene,
                         'sprrho_linear':sprrho,
                         'prsrho_linear':prsrho}))
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
    g = plt.scatter(predicted, measured, s=6, alpha=1, label=original)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plt.text(_REL_PLOT_MAX - 0.2, _REL_PLOT_MIN + 0.2, template.format(**vars()))
  plt.legend(loc='lower left', fontsize='xx-small')
  plt.tight_layout()
  plotfile = PLOTDIR / 'scatter.agg.{gene}.png'.format(**locals())
  plt.savefig(plotfile, dpi=_FIGDPI)
  plt.close()

rhoframe = pd.concat(rhos, axis=1).T
mapping_lib.make_mapping(rhoframe, 'gene', 'sprrho_linear', UNGD)
mapping_lib.make_mapping(rhoframe, 'gene', 'prsrho_linear', UNGD)
eval_lib.plot_confusion(data, PLOTDIR)
