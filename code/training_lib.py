#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import itertools
import logging
import pathlib
import random
import sys

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import AvgPool2D
from keras.layers import Dropout
from keras.models import Sequential

from sklearn import preprocessing as skpreproc

import mapping_lib
import gamma_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)


UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_CODEFILE = pathlib.Path(__file__).name
CONVNET_MODELDIR = (UNGD / _CODEFILE).with_suffix('.convnet.models')
LINEAR_MODELDIR = (UNGD / _CODEFILE).with_suffix('.linear.models')

_DEFAULT_LINEAR_HYPERPARAMS = dict()

_DEFAULT_NN_HYPERPARAMS = dict()
_DEFAULT_NN_HYPERPARAMS['first_conv_layer_nodes'] = 64
_DEFAULT_NN_HYPERPARAMS['second_conv_layer_nodes'] = 128
_DEFAULT_NN_HYPERPARAMS['first_dense_layer_nodes'] = 768
_DEFAULT_NN_HYPERPARAMS['second_dense_layer_nodes'] = 384

# TODO(jsh): DEBUG ONLY
# _DEFAULT_HYPERPARAMS['first_conv_layer_nodes'] = 64
# _DEFAULT_HYPERPARAMS['second_conv_layer_nodes'] = 128
# _DEFAULT_HYPERPARAMS['first_dense_layer_nodes'] = 64
# _DEFAULT_HYPERPARAMS['second_dense_layer_nodes'] = 32
# TODO(jsh): DEBUG ONLY

def build_conv_net_model(hyperparams=_DEFAULT_NN_HYPERPARAMS):
  # TODO(jsh): use dict.update to merge passed hyperparams, don't clobber
  model = Sequential()
  model.add(Conv2D(hyperparams['first_conv_layer_nodes'],
                   input_shape=(21,4,2), padding='same',
                   kernel_size=(4,3),
                   activation='relu'))
  model.add(MaxPool2D(pool_size=(2,1)))
  model.add(Conv2D(hyperparams['second_conv_layer_nodes'],
                   kernel_size=(2,2),
                   activation='relu'))
  model.add(MaxPool2D(pool_size=(2,1)))
  model.add(Flatten())
  model.add(Dense(hyperparams['first_dense_layer_nodes'],
                  activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(hyperparams['second_dense_layer_nodes'],
                  activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.compile(loss='mse', metrics=['mse'], optimizer='adam')
  return model

def build_linear_model(num_features, *, hyperparams=_DEFAULT_LINEAR_HYPERPARAMS):
  model = Sequential()
  # model.add(Dense(1, input_dim=num_features, activation='relu'))
  model.add(Dense(1, input_dim=num_features, activation='linear'))
  model.compile(loss='mse', metrics=['mse'], optimizer='adam')
  return model

def filter_for_training(variantframe, datadir):
  var_oneoff = mapping_lib.get_mapping('variant', 'is_oneoff', datadir)
  maskset = var_oneoff.loc[var_oneoff.is_oneoff].index
  oneoffs = variantframe.loc[variantframe.index.intersection(maskset)]
  return oneoffs

def downsample_families(data, ratio, datadir):
  var_orig = mapping_lib.get_mapping('variant', 'original', datadir)
  oanno = pd.merge(data, var_orig, left_on='variant', right_index=True)
  families = set(oanno.original.unique())
  samplesize = int(len(families) * ratio)
  sample = random.sample(families, samplesize)
  samplevariants = oanno.loc[oanno.original.isin(sample)].index
  littledata = data.loc[data.index.intersection(samplevariants)]
  return littledata

def one_hot_pair_encoder(datadir):
  var_orig = mapping_lib.get_mapping('variant', 'original', datadir)
  var_pam = mapping_lib.get_mapping('variant', 'pam', datadir)
  bases = ['A', 'C', 'G', 'T']
  enc = skpreproc.OneHotEncoder(categories=[bases], sparse=False)
  def encoder(seq):
    orig = var_orig.loc[seq].original
    pam = var_pam.loc[seq].pam
    varplus = seq + pam[0]
    origplus = orig + pam[0]
    V = np.array(list(varplus))
    V = V.reshape(len(varplus), 1)
    O = np.array(list(origplus))
    O = O.reshape(len(origplus), 1)
    onehot = np.stack([enc.fit_transform(V),
                       enc.fit_transform(O)], axis=-1)
    return ((varplus, origplus), onehot)
  return encoder


def expand_dummies(frame):
  categories = dict()
  bases = ['A', 'C', 'G', 'T', 'N']
  idxs = [x for x in range(20)]  # Magic number because guidelen is fixed.
  pairs = [''.join(pair) for pair in itertools.product(bases, bases)]
  combos = ['_'.join((pair, str(idx))) for pair, idx in itertools.product(pairs, idxs)]
  categories['mm_idx'] = idxs
  categories['mm_trans'] = pairs
  categories['mm_suffix'] = pairs
  categories['mm_prefix'] = pairs
  categories['mm_brackets'] = pairs
  categories['firstbase'] = bases
  categories['mm_both'] = combos
  widecols = list()
  for column in frame.columns:
    if column not in categories:
      continue
    frame[column] = frame[column].astype(CategoricalDtype(categories[column]))
  return pd.get_dummies(frame)


def feature_encoder(datadir):
  var_orig = mapping_lib.get_mapping('variant', 'original', datadir)
  var_pam = mapping_lib.get_mapping('variant', 'pam', datadir)
  def encoder(seq):
    orig = var_orig.loc[seq].original
    pam = var_pam.loc[seq].pam
    varplus = seq + pam[0]
    origplus = orig + pam[0]
    mm_idx = None
    for i in range(len(varplus)):
      if varplus[i] != origplus[i]:
        if origplus != varplus[:i] + origplus[i] + varplus[i+1:]:
          template = 'too many mismatches in pair {varplus} <- {origplus}'
          raise ValueError(template.format(**locals()))
        mm_idx = i
    if mm_idx == None:
      template = 'no mismatch in pair {varplus} <- {origplus}'
      raise ValueError(template.format(**locals()))
    features = dict()
    features['mm_idx'] = mm_idx
    mm_trans = ''.join([origplus[mm_idx], varplus[mm_idx]])
    features['mm_trans'] = mm_trans
    features['mm_both'] = '_'.join([str(mm_idx), mm_trans])
    features['gc_cont'] = origplus.count('G') + origplus.count('C')
    features['firstbase'] = origplus[0]
    wrapped = 'NN' + varplus + 'NN'
    features['mm_brackets'] = ''.join([wrapped[mm_idx+1], wrapped[mm_idx+3]])
    features['mm_prefix'] = wrapped[mm_idx:mm_idx+2]
    features['mm_suffix'] = wrapped[mm_idx+3:mm_idx+5]
    row = pd.Series(features)
    return row
  return encoder
