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

gamma_lib.compute_rough_gammas(UNGD)
gamma_lib.compute_dca_smooth_gammas(UNGD)
gamma_lib.compute_normed_gammas(UNGD)

gamma_lib.map_variant_to_mean_full_gamma(UNGD)
gamma_lib.map_variant_to_mean_full_gamma(UNGD, dose='low')
gamma_lib.map_variant_to_mean_full_gamma(UNGD, dose='high')
