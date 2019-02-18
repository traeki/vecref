#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

from keras.models import load_model
import numpy as np
import pandas as pd

import choice_lib
import mapping_lib

from predict_all_linear import PREDFILE as linear_preds

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
COMPS = UNGD / 'lib2_comps.tsv'
_CODEFILE = pathlib.Path(__file__).name
N_FAMILIES = 10
OLD_GUIDES_PER_LOCUS = 10
NEW_GUIDES_PER_FAMILY = 10

# read in measured values for all applicable relgammas
var_orig = mapping_lib.get_mapping('variant', 'original', UNGD)
var_loc = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
measured = mapping_lib.get_mapping('variant', 'relgamma', UNGD)
measured.reset_index(inplace=True)
measured['original'] = measured.variant.map(var_orig.original)
measured['locus_tag'] = measured.variant.map(var_loc.locus_tag)
# read in identified comparison set and comprehensive prediction set
preds = pd.read_csv(linear_preds, sep='\t')
predset = set((x.variant, x.original, x.locus_tag) for (_,x) in preds.iterrows())
comps = pd.read_csv(COMPS, sep='\t')
compset = set((x.variant, x.original, x.locus_tag) for (_,x) in comps.iterrows())
# actually loop over locus tags and choose measure
old_guides = dict()
new_guides = dict()
for lt in preds.locus_tag.unique():
  logging.info('Examining options for locus_tag: {lt}...'.format(**locals()))
  lt_comps = comps.loc[comps.locus_tag == lt]
  lt_preds = preds.loc[preds.locus_tag == lt]
  lt_measured = measured.loc[measured.locus_tag == lt]
  lt_parents = choice_lib.pick_n_parents(lt_comps,
                                         lt_preds,
                                         N_FAMILIES)
  old_guides[lt] = choice_lib.choose_n_meas(measured,
                                            OLD_GUIDES_PER_FAMILY)
  new_guides[lt] = choice_lib.choose_n_for_each(lt_parents,
                                                lt_preds,
                                                lt_measured,
                                                NEW_GUIDES_PER_FAMILY)


import IPython; IPython.embed()
# TODO(jsh): Select (2 + 8) parents
# TODO(jsh): Select 2 * 5 measured guides
# TODO(jsh): Select 2 * 10 unmeasured guides
# TODO(jsh): Select 8 * 10 guides

# Choose two parents from each gene that we've seen before.
# ... Choose 8 never seen before.
# ... Choose enough seen before to get to 10 total

# NOTE(jsh): Manually grab parents from original set
# NOTE(jsh):    - verify there are 2x
# NOTE(jsh): Grab remaining parents at random
# NOTE(jsh): Keep parents as a multiset/list
# NOTE(jsh): Select children by broad bins
# NOTE(jsh): Fall back to "non-terminal"
# NOTE(jsh): Fall back to "broken"
# NOTE(jsh): Store per-gene guides in a set, and ... ?
