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
import seaborn as sns
from sklearn import preprocessing as skpreproc

import eval_lib
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
encoder = training_lib.get_linear_encoder()
X, y = eval_lib.featurize_training_data(encoder, data, UNGD)
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

locusmap = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
genemap = mapping_lib.get_mapping('locus_tag', 'gene_name', UNGD)
geneids = genemap.loc[locusmap.loc[data.index].locus_tag]
geneids.index = data.index
data['y_pred'] = cross_predictions
data['gene_name'] = geneids
data['original'] = familymap.original
data['dose'] = 'sober'
lowrg = mapping_lib.get_mapping('variant', 'relgamma', UNGD, dose='low')
lowcopy = data.copy()
lowcopy.dose = 'low'
lowcopy.relgamma = lowrg
highrg = mapping_lib.get_mapping('variant', 'relgamma', UNGD, dose='high')
highcopy = data.copy()
highcopy.dose = 'high'
highcopy.relgamma = highrg
sobercopy = data.copy()
data = pd.concat([sobercopy, lowcopy, highcopy])
data['y_meas'] = data.relgamma

for (gene, original), group in data.groupby(['gene_name', 'original']):
  group = group.reset_index()
  predicted = group.y_pred
  measured = group.y_meas
  fig, ax = plt.subplots(1, 1, figsize=(6,6))
  template = 'Predictions vs. Measurements by Dose\n{gene}<-{original}'
  main_title_str = template.format(**locals())
  plt.title(main_title_str)
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.xlabel('Predicted relative γ')
  plt.ylabel('Measured relative γ')
  predicted = group.y_pred
  measured = group.y_meas
  ax = sns.scatterplot(ax=ax,
                       data=group,
                       legend=False,
                       x='y_pred', y='y_meas', hue='variant',
                       size='dose', sizes={'sober':30,'low':20,'high':10},
                       alpha=1,
                       s=20)
  plt.gca().invert_xaxis()
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plotfile = PLOTDIR / 'scatter.dose.{gene}.{original}.png'.format(**locals())
  plt.savefig(plotfile, dpi=_FIGDPI)
  plt.close()
