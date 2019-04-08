#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

from keras.models import load_model
import numpy as np
import pandas as pd

import choice_lib
import eval_lib
import training_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_CODEFILE = pathlib.Path(__file__).name

BSU_PREDFILE = (UNGD / _CODEFILE).with_suffix('.bsu.tsv')
BSU_BASE = pathlib.Path('/home/jsh/gd/proj/lowficrispri/docs/20180626_rebase/')
BSU_TARGETS = BSU_BASE / 'data/bsu.NC_000964.targets.all.tsv'
BSU_LOCI = BSU_BASE / 'output/choose_important_genes.tsv'

ECO_PREDFILE = (UNGD / _CODEFILE).with_suffix('.eco.tsv')
ECO_BASE = pathlib.Path('/home/jsh/gd/genomes/eco_bw25113/')
ECO_TARGETS = ECO_BASE / 'eco_bw25113.merged.gb.targets.all.tsv'
ECO_LOCI = ECO_BASE / 'ms_loci.tsv'

if __name__ == '__main__':
  # Load the trained model
  model = load_model(str(training_lib.LINEAR_MODELDIR / 'model.max.d5'))
  traindata = eval_lib.fetch_training_data(UNGD)
  encoder = training_lib.get_linear_encoder()
  trainX, trainy = eval_lib.featurize_training_data(encoder, traindata, UNGD)
  X_scaler, y_scaler, _, _ = eval_lib.scale_training_data_linear(trainX, trainy)

  # Get all relevan parent/child pairs
  eco_set = choice_lib.build_and_filter_pairs(
      targetfile=ECO_TARGETS,
      locustagfile=ECO_LOCI)
  bsu_set = choice_lib.build_and_filter_pairs(
      targetfile=BSU_TARGETS,
      locustagfile=BSU_LOCI)

  # prep pairs for model
  eco_encodings = eco_set.apply(encoder, axis=1)
  eco_Xframe = eco_encodings.set_index(eco_set.variant)
  eco_Xframe = training_lib.expand_dummies(eco_Xframe)
  eco_X = np.array(eco_Xframe, dtype=float)
  eco_X = X_scaler.transform(eco_X)
  bsu_encodings = bsu_set.apply(encoder, axis=1)
  bsu_Xframe = bsu_encodings.set_index(bsu_set.variant)
  bsu_Xframe = training_lib.expand_dummies(bsu_Xframe)
  bsu_X = np.array(bsu_Xframe, dtype=float)
  bsu_X = X_scaler.transform(bsu_X)

  # score pairs
  eco_raw_predictions = model.predict(eco_X)
  eco_y_pred = y_scaler.inverse_transform(eco_raw_predictions)
  eco_set['y_pred'] = eco_y_pred
  eco_set.to_csv(ECO_PREDFILE, sep='\t', index=False)
  bsu_raw_predictions = model.predict(bsu_X)
  bsu_y_pred = y_scaler.inverse_transform(bsu_raw_predictions)
  bsu_set['y_pred'] = bsu_y_pred
  bsu_set.to_csv(BSU_PREDFILE, sep='\t', index=False)
