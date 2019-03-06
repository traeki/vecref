#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import training_lib
import mapping_lib

from choose_v2_guides import NEWGUIDEFILE as guidefile

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_CODEFILE = pathlib.Path(__file__).name
MM_HEATMAP = (UNGD / _CODEFILE).with_suffix('.mm.png')
BASE_HEATMAP = (UNGD / _CODEFILE).with_suffix('.base.png')
_PLOT_DPI = 600

def plot_presence(tallyframe, xlabel, ylabel, plotfile):
  logging.info('Drawing presence heatmap to {plotfile}...'.format(**vars()))
  plt.figure(figsize=(10,6))
  expected = tallyframe.sum().sum() / (tallyframe.shape[0] * tallyframe.shape[1])
  ax = sns.heatmap(tallyframe, vmin=0, vmax=expected, cmap='inferno')
  main_title_str = 'Presence Tally [{xlabel} X {ylabel}]'.format(**locals())
  plt.title(main_title_str)
  plt.tight_layout()
  plt.savefig(plotfile, dpi=_PLOT_DPI)
  plt.clf()

if __name__ == '__main__':
  logging.info('Reading preds from {guidefile}...'.format(**locals()))
  guides = pd.read_csv(guidefile, sep='\t')
  encoder = training_lib.get_linear_encoder()
  encodings = guides.apply(encoder, axis=1)
  data = pd.concat([guides, encodings], axis=1)
  mm_tally = data.groupby(['mm_idx', 'mm_trans']).count().pam
  mm_tally = mm_tally.unstack().T
  basegrid = pd.DataFrame(np.array([list(x) for x in data.original]))
  base_tally = [basegrid[i].value_counts() for i in basegrid.columns]
  base_tally = pd.DataFrame(base_tally).T
  logging.info('MM_TALLY:\n{mm_tally}'.format(**locals()))
  logging.info('BASE_TALLY:\n{base_tally}'.format(**locals()))
  plot_presence(mm_tally, 'index', 'trans', MM_HEATMAP)
  plot_presence(base_tally, 'position', 'base', BASE_HEATMAP)
