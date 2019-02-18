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
PREDFILE = (UNGD / _CODEFILE).with_suffix('.tsv')

if __name__ == '__main__':
  # Load the trained model
  model = load_model(str(training_lib.LINEAR_MODELDIR / 'model.max.d5'))
  traindata = eval_lib.fetch_training_data(UNGD)
  encoder = training_lib.get_linear_encoder()
  trainX, trainy = eval_lib.featurize_training_data(encoder, traindata, UNGD)
  X_scaler, y_scaler, _, _ = eval_lib.scale_training_data_linear(trainX, trainy)

  # Get all relevan parent/child pairs
  chosen = choice_lib.build_and_filter_pairs()

  # prep pairs for model
  encodings = chosen.apply(encoder, axis=1)
  Xframe = encodings.set_index(chosen.variant)
  Xframe = training_lib.expand_dummies(Xframe)
  X = np.array(Xframe, dtype=float)
  X = X_scaler.transform(X)

  # score pairs
  raw_predictions = model.predict(X)
  y_pred = y_scaler.inverse_transform(raw_predictions)
  chosen['y_pred'] = y_pred

  chosen.to_csv(PREDFILE, sep='\t', index=False)
