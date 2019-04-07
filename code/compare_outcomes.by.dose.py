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

_PLOT_MIN = -1.2
_PLOT_MAX = 0.2

_FIGDPI = 300

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

############################
# Re-load/process raw data #
############################
sober = mapping_lib.get_mapping('variant', 'gamma', UNGD, dose='sober')
sober.columns = ['sober']
low = mapping_lib.get_mapping('variant', 'gamma', UNGD, dose='low')
low.columns = ['low']
high = mapping_lib.get_mapping('variant', 'gamma', UNGD, dose='high')
high.columns = ['high']
familymap = mapping_lib.get_mapping('variant', 'original', UNGD)

locusmap = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
genemap = mapping_lib.get_mapping('locus_tag', 'gene_name', UNGD)
geneids = genemap.loc[locusmap.locus_tag]
geneids.index = locusmap.index

data = pd.concat([familymap, sober, low, high, geneids], axis=1, sort=True)

for gene, group in data.groupby(['gene_name']):
  group = group.reset_index()
  fig, ax = plt.subplots(1, 1, figsize=(6,6))
  template = 'Impact of Trimethoprim on gamma\n{gene}'
  main_title_str = template.format(**locals())
  plt.title(main_title_str)
  plt.xlim(_PLOT_MIN, _PLOT_MAX)
  plt.ylim(_PLOT_MIN, _PLOT_MAX)
  plt.scatter(group.sober, group.low,
              alpha=0.5, color='xkcd:blue', s=4, label='low')
  plt.scatter(group.sober, group.high,
              alpha=0.5, color='xkcd:orange', s=1.5, label='high')
  plt.plot([-1.18, 0.18], [-1.18, 0.18], lw=1, linestyle='--', color='xkcd:green')
  plt.plot([0, 0], [-1.18, 0.18], lw=1, linestyle='--', color='xkcd:red')
  plt.xlabel('γ [no drug]')
  plt.ylabel('γ [drug]')
  plt.legend()
  plt.tight_layout()
  plotfile = PLOTDIR / 'scatter.trimpact.{gene}.png'.format(**locals())
  plt.savefig(plotfile, dpi=_FIGDPI)
  plt.close()
