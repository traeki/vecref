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

DOSE = 'sober'
gammas = gamma_lib.get_normed_gammas(UNGD)
gammas = gamma_lib.select_dose(gammas, DOSE, UNGD)
gammas = gammas.stack(level='sid', dropna=False)[['03']].unstack()

parent_gammas = gamma_lib.get_parent_gammas(UNGD)
parent_gammas = gamma_lib.select_dose(parent_gammas, DOSE, UNGD)
parent_gammas = parent_gammas.stack(level='sid', dropna=False)[['03']].unstack()
parent_gammas.sort_index(axis=1, inplace=True)

gammas = gammas.stack(level='span').reset_index()
gammas = gammas.drop('span', axis=1).set_index('variant')

shutil.rmtree(PLOTDIR, ignore_errors=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6,6))
template = 'Replicate 1 vs. Replicate 2'
main_title_str = template.format(**locals())
plt.title(main_title_str)
plt.xlim(_REL_PLOT_MIN, _REL_PLOT_MAX)
plt.ylim(_REL_PLOT_MIN, _REL_PLOT_MAX)
g = sns.scatterplot(data=gammas, x='d1a', y='d1b', alpha=0.2, s=6)
plt.xlabel('Fitness in Replicate 1')
plt.ylabel('Fitness in Replicate 2')
plt.tight_layout()
plotfile = PLOTDIR / 'replicate.comp.png'.format(**locals())
plt.savefig(plotfile, dpi=_FIGDPI)
plt.close()

plt.figure(figsize=(6,6))
template = 'Fitness in "Micropool" vs Full Pool'
x = [-.348, -.273, -.204, -.172, -.071, -.053, -.019]
y = [-.578, -.496, -.439, -.341, -.217, -.106, -.022]
main_title_str = template.format(**locals())
plt.title(main_title_str)
plt.xlim(-.6, 0)
plt.ylim(-.6, 0)
g = sns.scatterplot(x=x, y=y, alpha=1, s=15)
plt.xlabel('Fitness in Full Pool')
plt.ylabel('Fitness in "Micropool"')
plt.tight_layout()
plotfile = PLOTDIR / 'micropool.validation.png'.format(**locals())
plt.savefig(plotfile, dpi=_FIGDPI)
plt.close()
