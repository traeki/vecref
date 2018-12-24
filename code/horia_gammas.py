#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

import mapping_lib
import gamma_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
DIR_PREFIX = pathlib.Path(__file__).parents[1]
_HORIA_FILE = 'horia.gammas.tsv'

pg = gamma_lib.get_parent_gammas(UNGD)
cg = gamma_lib.get_child_gammas(UNGD)
rg = gamma_lib.unfiltered_mean_relgammas(UNGD)
oo = mapping_lib.get_mapping('variant', 'is_oneoff', UNGD)
oi = oo.loc[oo.is_oneoff].index
mpg = pg.stack(level=0, dropna=False)['03'].unstack().mean(axis=1)
mcg = cg.stack(level=0, dropna=False)['03'].unstack().mean(axis=1)
import IPython; IPython.embed()
rg['parent_gamma'] = mpg
rg['child_gamma'] = mcg
rg.loc[oi].to_csv(UNGD / _HORIA_FILE, sep='\t')
