#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

import mapping_lib
import gamma_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
DIR_PREFIX = pathlib.Path(__file__).parents[1]
_CODEFILE = pathlib.Path(__file__).name
PLOTDIR = (UNGD / _CODEFILE).with_suffix('.plots')

_REL_PLOT_MIN = -1.2
_REL_PLOT_MAX = 0.2

_FIGDPI = 300

exp_taggers = list()
exp_taggers.append(('span01', lambda s: (s[-2:] == '01')))
exp_taggers.append(('span02', lambda s: (s[-2:] == '02')))
exp_taggers.append(('span03', lambda s: (s[-2:] == '03')))
exp_taggers.append(('span12', lambda s: (s[-2:] == '12')))
exp_taggers.append(('span13', lambda s: (s[-2:] == '13')))
exp_taggers.append(('span23', lambda s: (s[-2:] == '23')))
exp_taggers.append(('none', lambda s: (s[2] == 'a' or s[:2] == 'd1')))
exp_taggers.append(('low', lambda s: (s[2] != 'a' and s[:2] == 'd2')))
exp_taggers.append(('high', lambda s: (s[2] != 'a' and s[:2] == 'd3')))

all_taggers = list()
all_taggers.append(('span01', lambda s: (s[-2:] == '01')))
all_taggers.append(('span02', lambda s: (s[-2:] == '02')))
all_taggers.append(('span03', lambda s: (s[-2:] == '03')))
all_taggers.append(('span12', lambda s: (s[-2:] == '12')))
all_taggers.append(('span13', lambda s: (s[-2:] == '13')))
all_taggers.append(('span23', lambda s: (s[-2:] == '23')))
all_taggers.append(('none', lambda s: (s[2] == 'a' or s[:2] == 'd1')))
all_taggers.append(('low', lambda s: (s[2] != 'a' and s[:2] == 'd2')))
all_taggers.append(('high', lambda s: (s[2] != 'a' and s[:2] == 'd3')))
all_taggers.append(('a', lambda s: (s[2] == 'a')))
all_taggers.append(('b', lambda s: (s[2] == 'b')))
all_taggers.append(('c', lambda s: (s[2] == 'c')))
all_taggers.append(('d1', lambda s: (s[:2] == 'd1')))
all_taggers.append(('d2', lambda s: (s[:2] == 'd2')))
all_taggers.append(('d3', lambda s: (s[:2] == 'd3')))

X = gamma_lib.get_rough_gammas(UNGD)
A = np.ma.masked_invalid(X)

D = np.asarray([[tagger(col) for (_, tagger) in all_taggers] for col in X])
D_span = gamma_lib.rebase(A, D)
XDDt = pd.DataFrame(D_span, index=X.index, columns=X.columns).copy()
XDDt.dropna(inplace=True, axis='index')
smoothed = XDDt[[col for col in XDDt.columns if col.endswith('03')]]

E = np.asarray([[tagger(col) for (_, tagger) in exp_taggers] for col in X])
E_span = gamma_lib.rebase(A, E)
XEEt = pd.DataFrame(E_span, index=X.index, columns=X.columns)
XNNt = X - XEEt
XNNt.dropna(inplace=True, axis='index')
U_noise, s_noise, Vt_noise = np.linalg.svd(XNNt, full_matrices=True)

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

Z_CUTOFF = 3
MAX_DIM = 1

for i in range(MAX_DIM):
  scoredim = i
  score = U_noise[scoredim]
  score = (score - score.mean()) / score.std()
  hits = (score < -Z_CUTOFF).astype(np.uint8)
  data = smoothed
  data = data[[col for col in data.columns if col[:2] == 'd1' or col[2] == 'a']]
  data = data.copy()
  data['hits'] = hits

  plt.figure(figsize=(6,6))
  template = 'Replicate comparisons tagged for Noise[{i}]'
  main_title_str = template.format(**locals())
  plt.title(main_title_str)
  plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
  g = sns.pairplot(data=data,
                   hue='hits',
                   diag_kind='hist',
                   plot_kws={'s':3,
                             'linewidth':0})
  plt.tight_layout()
  plotfile = PLOTDIR / 'pairwise.noise.{i}.png'.format(**locals())
  plt.savefig(plotfile, dpi=_FIGDPI)
  plt.close()
