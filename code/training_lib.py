#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import random
import sys

import numpy as np
import pandas as pd

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


_DEFAULT_HYPERPARAMS = dict()
_DEFAULT_HYPERPARAMS['first_conv_layer_nodes'] = 64
_DEFAULT_HYPERPARAMS['second_conv_layer_nodes'] = 128
_DEFAULT_HYPERPARAMS['first_dense_layer_nodes'] = 768
_DEFAULT_HYPERPARAMS['second_dense_layer_nodes'] = 384

def build_conv_net_model(hyperparams=_DEFAULT_HYPERPARAMS):
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
