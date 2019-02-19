#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import choice_lib
import mapping_lib

from choose_v2_guides import OLDGUIDEFILE
from choose_v2_guides import NEWGUIDEFILE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_CODEFILE = pathlib.Path(__file__).name
OLDPLOT = (UNGD / _CODEFILE).with_suffix('.old.png')
NEWPLOT = (UNGD / _CODEFILE).with_suffix('.new.png')

_REL_PLOT_MIN = -1.2
_REL_PLOT_MAX = 0.2
_FIGDPI = 300

if __name__ == '__main__':
  logging.info('Reading guide files from ...'.format(**locals()))
  logging.info('{OLDGUIDEFILE}...'.format(**locals()))
  oldguides = pd.read_csv(OLDGUIDEFILE, sep='\t')
  logging.info('{NEWGUIDEFILE}...'.format(**locals()))
  newguides = pd.read_csv(NEWGUIDEFILE, sep='\t')

  logging.info('Building "old" plot -> {OLDPLOT}'.format(**locals()))
  plt.figure(figsize=(10,6))
  ax = sns.stripplot(x='locus_tag', y='relgamma', hue='original',
                     data=oldguides, jitter=False, size=3)
  ax.legend().remove()
  ax.set(xticks=[])
  ax.set_xlabel('')
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.gca().invert_yaxis()
  plt.title('V2 Lib: Subset of Old Guides'.format(**locals()))
  plt.tight_layout()
  plt.savefig(OLDPLOT, dpi=_FIGDPI)

  plt.clf()
  plt.close()

  logging.info('Building "new" plot -> {NEWPLOT}'.format(**locals()))
  plt.figure(figsize=(10,6))
  ax = sns.stripplot(x='locus_tag', y='y_pred', hue='original',
                     data=newguides, jitter=False, size=2)
  ax.legend().remove()
  ax.set(xticks=[])
  ax.set_xlabel('')
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.gca().invert_yaxis()
  plt.title('V2 Lib: Novel Predicted Guides'.format(**locals()))
  plt.tight_layout()
  plt.savefig(NEWPLOT, dpi=_FIGDPI)

  plt.clf()
  plt.close()
