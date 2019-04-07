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

# compute parent gammas
gamma_lib.derive_child_parent_gammas(UNGD)

# compute relative gammas
gamma_lib.map_variant_to_mean_full_relative_gamma(UNGD, dose='sober')
gamma_lib.map_variant_to_mean_full_relative_gamma(UNGD, dose='low')
gamma_lib.map_variant_to_mean_full_relative_gamma(UNGD, dose='high')
gamma_lib.map_variant_to_mean_full_relative_gamma(UNGD, filtered=False)

# break out relgamma bins
gamma_lib.map_variant_to_bin(UNGD)
