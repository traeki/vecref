#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import pickle
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from keras.models import load_model

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
_REL_PLOT_MAX = 0.5

############################
# Re-load/process raw data #
############################
data = mapping_lib.get_mapping('variant', 'relgamma', UNGD)
data = training_lib.filter_for_training(data, UNGD)
data = data.dropna()
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)
familymap = familymap.loc[data.index]
encoder = training_lib.one_hot_pair_encoder(UNGD)
X = np.stack([encoder(x)[1] for x in data.index], axis=0)
y = np.array(data.relgamma)
cross_predictions = np.full_like(y, np.nan)

########################
# Read Prediction Data #
########################
modeldir = training_lib.CONVNET_MODELDIR
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
  test = testmask.nonzero()[0]
  train = (~testmask).nonzero()[0]
  test_predictions = model.predict(X[test]).ravel()
  train_predictions = model.predict(X[train]).ravel()
  cross_predictions[test] = test_predictions

  plt.figure(figsize=(6,6))
  plt.scatter(test_predictions, y[test], marker='.', alpha=.2, label='test')
  plt.scatter(train_predictions, y[train], marker='.', alpha=.2, label='train')
  plt.title('Model Predictions [Fold {i}]'.format(**locals()))
  plt.xlabel('predicted')
  plt.ylabel('measured')
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plotfile = PLOTDIR / 'scatter.{i}.png'.format(**locals())
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
plotfile = PLOTDIR / 'scatter.agg.png'.format(**locals())
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
  plotfile = PLOTDIR / 'scatter.agg.{gene}.png'.format(**locals())
  plt.savefig(plotfile)
  plt.close()

eval_lib.plot_confusion(data, PLOTDIR)