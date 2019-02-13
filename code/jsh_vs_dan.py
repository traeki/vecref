#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import operator
import pathlib
import pickle
import shutil

import keras
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
DANDIR = UNGD / 'dancomp'
_DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name
PLOTDIR = (UNGD / _CODEFILE).with_suffix('.plots')
MODELDIR = (UNGD / _CODEFILE).with_suffix('.models')

_DATA_FRACTION = 1
_K_FOLD_SPLITS = 3
_REL_PLOT_MIN = -0.2
_REL_PLOT_MAX = 2
_BATCH_SIZE = 32
_EPOCHS = 10
_FIGDPI = 300
_USE_SCALING = False


shutil.rmtree(MODELDIR, ignore_errors=True)
MODELDIR.mkdir(parents=True, exist_ok=True)
shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

# Read in the model structures
jsons = dict()
jsons['jsh'] = open(DANDIR / 'jsh_model.json', 'r').read()
jsons['old'] = open(DANDIR / 'old_model.json', 'r').read()

# Load in the data files
Xs = dict()
Xs['jsh'] = np.load(DANDIR / 'X_jsh.npy')
Xs['old'] = np.load(DANDIR / 'X_18925,21,4,2.npy')
ys = dict()
ys['jsh'] = np.load(DANDIR / 'y_jsh.npy')
ys['jsh'] += 1  # NOTE correct for offset NOTE
ys['old'] = np.load(DANDIR / 'y_18925.npy')
bases = ['A', 'C', 'G', 'T']
encoded = Xs['old']
indexed = np.argmax(encoded, 2)
decpairs = list()
for pair in indexed:
  ori = pair[:,0]
  var = pair[:,1]
  ori = ''.join([bases[idx] for idx in ori])
  var = ''.join([bases[idx] for idx in var])
  decpairs.append(pd.Series([ori, var], index=['original', 'variant']))
decoded = pd.concat(decpairs, axis=1).T

tags = dict()
tags['jsh'] = pd.read_csv(DANDIR / 'tags_jsh.tsv', sep='\t')
tags['old'] = decoded[['original']]

def n_mismatched(row):
  ori = row.original
  var = row.variant
  pairs = zip(ori, var)
  bools = [o != v for o, v in pairs]
  n = sum(bools)
  return n
# counts = decoded.apply(n_mismatched, axis=1)
# keep = counts.loc[counts == 1].index
# Xs['old'] = Xs['old'][keep]
# ys['old'] = ys['old'][keep]
# tags['old'] = tags['old'].loc[keep]

jsh_old_ratio = len(Xs['jsh'])/len(Xs['old'])
epochs = dict()
epochs['jsh'] = int(_EPOCHS / jsh_old_ratio)
epochs['old'] = int(_EPOCHS)

# ...For each model / dataset / split
for jname in jsons:
  if jname == 'jsh':
    continue # We're convinced the model doesn't matter, for now
  for source in Xs:
    kfs = dict()
    flat = skmodsel.KFold(_K_FOLD_SPLITS, shuffle=True)
    kfs['flat'] = flat.split(Xs[source], ys[source])
    grouped = skmodsel.GroupKFold(_K_FOLD_SPLITS)
    kfs['grouped'] = grouped.split(Xs[source], ys[source], tags[source])
    for kfname, kf in kfs.items():
      for fold, (train, test) in enumerate(kf):
        template = "Training '{jname}' model on {source}/{kfname}[{fold}]"
        logging.info(template.format(**locals()))
        # ...Train model on split
        json = jsons[jname]
        X = Xs[source]
        y = ys[source]
        y_orig = y
        if _USE_SCALING:
          X_scaler = dict()
          for i in range(X.shape[1]):
            for j in range(X.shape[3]):
              X_scaler[(i,j)] = skpreproc.StandardScaler()
              X[:,i,:,j] = X_scaler[(i,j)].fit_transform(X[:,i,:,j])
          y_scaler = skpreproc.StandardScaler()
          y = y_scaler.fit_transform(y)
        model = keras.models.model_from_json(json)
        model.compile(loss='mse', metrics=['mse'], optimizer='adam')
        model_history = model.fit(X[train], y[train],
                                  batch_size=_BATCH_SIZE,
                                  epochs=epochs[source],
                                  validation_data=(X[test], y[test]))
        template = '{0}.{jname}.src.{source}.{fold}.{kfname}.{1}'
        modelfile = MODELDIR / template.format('model', 'd5', **locals())
        modelfile = str(modelfile)
        model.save(modelfile)
        np.save(MODELDIR / template.format('split', 'train.npy', **locals()),
                train)
        np.save(MODELDIR / template.format('split', 'test.npy', **locals()),
                test)
        train_predictions = model.predict(X[train])
        if _USE_SCALING:
          train_predictions = y_scaler.inverse_transform(train_predictions)
        train_predictions = train_predictions.reshape(-1,1)
        test_predictions = model.predict(X[test])
        if _USE_SCALING:
          test_predictions = y_scaler.inverse_transform(test_predictions)
        test_predictions = test_predictions.reshape(-1,1)

        # ...Plot error over time
        plt.figure(figsize=(6,6))
        plt.plot(model_history.history['mean_squared_error'])
        plt.plot(model_history.history['val_mean_squared_error'])
        plt.title('Model Error [Fold {fold}]'.format(**locals()))
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.ylim(-0.2, 1.2)
        plt.legend(['Train', 'Test'], loc='upper right')
        plotfile = PLOTDIR / template.format('history', 'png', **locals())
        plt.savefig(plotfile, dpi=_FIGDPI)
        plt.close()

        # ...Plot overall outcomes, train vs test
        plt.figure(figsize=(6,6))
        # Plot and Pearson-score train
        pred = train_predictions.ravel()
        meas = y_orig[train].ravel()
        train_prs, _ = st.pearsonr(pred, meas)
        plt.scatter(pred, meas, marker='.', alpha=.2, label='train')
        # Plot and Pearson-score test
        pred = test_predictions.ravel()
        meas = y_orig[test].ravel()
        test_prs, _ = st.pearsonr(pred, meas)
        plt.scatter(pred, meas, marker='.', alpha=.2, label='test')
        # Annotate plot
        statstring = 'Train: {train_prs:.2f}\nTest: {test_prs:.2f}'
        plt.text(_REL_PLOT_MAX - 0.2, _REL_PLOT_MIN + 0.2,
                 statstring.format(**locals()))
        title = 'Predictions[{source}] vs. Measurements\n[{kfname}/{fold}]'
        plt.title(title.format(**locals()))
        plt.xlabel('predicted')
        plt.ylabel('measured')
        plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
        plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.legend(loc='lower left', fontsize='small')
        plt.tight_layout()
        plotfile = PLOTDIR / template.format('scatter', 'png', **locals())
        plt.savefig(plotfile, dpi=_FIGDPI)
        plt.close()


# ...For each model / dataset
for jname in jsons:
  for source in Xs:
    for target in Xs:
      if source == target:
        continue
      # ...Train model on source
      template = "Training '{jname}' model on {source}"
      logging.info(template.format(**locals()))
      json = jsons[jname]
      X = Xs[source]
      y = ys[source]
      y_orig = y
      if _USE_SCALING:
        X_scaler = dict()
        for i in range(X.shape[1]):
          for j in range(X.shape[3]):
            X_scaler[(i,j)] = skpreproc.StandardScaler()
            X[:,i,:,j] = X_scaler[(i,j)].fit_transform(X[:,i,:,j])
        y_scaler = skpreproc.StandardScaler()
        y = y_scaler.fit_transform(y)
      model = keras.models.model_from_json(json)
      model.compile(loss='mse', metrics=['mse'], optimizer='adam')
      model_history = model.fit(X, y,
                                batch_size=_BATCH_SIZE,
                                epochs=epochs[source])
      template = '{0}.{jname}.{source}.on.{target}.{1}'
      modelfile = MODELDIR / template.format('model', 'd5', **locals())
      modelfile = str(modelfile)
      model.save(modelfile)
      train_predictions = model.predict(X)
      if _USE_SCALING:
        train_predictions = y_scaler.inverse_transform(train_predictions)
      train_predictions = train_predictions.reshape(-1,1)
      test_predictions = model.predict(Xs[target])
      if _USE_SCALING:
        test_predictions = y_scaler.inverse_transform(test_predictions)
      test_predictions = test_predictions.reshape(-1,1)

      # ...Plot overall outcomes, train vs test
      plt.figure(figsize=(6,6))
      # Plot and Pearson-score train
      pred = train_predictions.ravel()
      meas = y_orig.ravel()
      train_prs, _ = st.pearsonr(pred, meas)
      plt.scatter(pred, meas, marker='.', alpha=.2, label='train')
      # Plot and Pearson-score test
      pred = test_predictions.ravel()
      meas = ys[target].ravel()
      test_prs, _ = st.pearsonr(pred, meas)
      plt.scatter(pred, meas, marker='.', alpha=.2, label='test')
      # Annotate plot
      statstring = 'Train: {train_prs:.2f}\nTest: {test_prs:.2f}'
      plt.text(_REL_PLOT_MAX - 0.2, _REL_PLOT_MIN + 0.2,
               statstring.format(**locals()))
      title = 'Predictions[{source}] vs. Measurements[{target}]'
      plt.title(title.format(**locals()))
      plt.xlabel('predicted')
      plt.ylabel('measured')
      plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
      plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
      plt.gca().invert_xaxis()
      plt.gca().invert_yaxis()
      plt.legend(loc='lower left', fontsize='small')
      plt.tight_layout()
      plotfile = PLOTDIR / template.format('scatter', 'png', **locals())
      plt.savefig(plotfile, dpi=_FIGDPI)
      plt.close()




# NOTE REFERENCE CODE BELOW NOTE #
# NOTE REFERENCE CODE BELOW NOTE #
# NOTE REFERENCE CODE BELOW NOTE #

# plt.figure(figsize=(6,6))
# plt.scatter(cross_predictions, y, marker='.', alpha=.2, label='X-prediction')
# plt.title('Predictions vs. Measurements\n[agg]'.format(**locals()))
# plt.xlabel('Predicted relative γ')
# plt.ylabel('Measured relative γ')
# plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
# plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.legend(loc='lower left', fontsize='small')
# plt.tight_layout()
# plotfile = PLOTDIR / 'scatter.agg.png'.format(**locals())
# plt.savefig(plotfile, dpi=_FIGDPI)
# plt.close()
#
# locusmap = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
# genemap = mapping_lib.get_mapping('locus_tag', 'gene_name', UNGD)
# geneids = genemap.loc[locusmap.loc[data.index].locus_tag]
# geneids.index = data.index
# data['y_meas'] = data.relgamma
# data['y_pred'] = cross_predictions
# data['gene_name'] = geneids
# data['original'] = familymap.original
#
# mapping_lib.make_mapping(data.reset_index(), 'variant', 'y_pred', UNGD)
#
# for gene, group in data.groupby('gene_name'):
#   predicted = group.y_pred
#   measured = group.y_meas
#   sprrho, _ = st.spearmanr(predicted, measured)
#   prsrho, _ = st.pearsonr(predicted, measured)
#   plt.figure(figsize=(6,6))
#   template = 'Predictions vs. Measurements\n{gene}'
#   main_title_str = template.format(**locals())
#   plt.title(main_title_str)
#   plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
#   plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
#   plt.xlabel('Predicted relative γ')
#   plt.ylabel('Measured relative γ')
#   template = 'Spearman: {sprrho:.2f}\nPearson: {prsrho:.2f}'
#   subgrouper = group.groupby('original')
#   for original, subgroup in subgrouper:
#     predicted = subgroup.y_pred
#     measured = subgroup.y_meas
#     g = plt.scatter(predicted, measured, s=6, alpha=1, label=original)
#   plt.gca().invert_xaxis()
#   plt.gca().invert_yaxis()
#   plt.text(_REL_PLOT_MAX - 0.2, _REL_PLOT_MIN + 0.2, template.format(**vars()))
#   plt.legend(loc='lower left', fontsize='xx-small')
#   plt.tight_layout()
#   plotfile = PLOTDIR / 'scatter.agg.{gene}.png'.format(**locals())
#   plt.savefig(plotfile, dpi=_FIGDPI)
#   plt.close()
#
# eval_lib.plot_confusion(data, PLOTDIR)
