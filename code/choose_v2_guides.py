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

if __name__ == '__main__':
  logging.info('Reading preds from {linear_preds}...'.format(**locals()))
  preds = pd.read_csv(linear_preds, sep='\t')
  logging.info('Reading comps from {COMPS}...'.format(**locals()))
  comps = pd.read_csv(COMPS, sep='\t')
  var_rg = mapping_lib.get_mapping('variant', 'relgamma', UNGD)
  comps['relgamma'] = comps.variant.map(var_rg.relgamma)
  # actually loop over locus tags and choose measure
  old_guides = dict()
  new_guides = dict()
  for lt in preds.locus_tag.unique():
    logging.info('Examining options for locus_tag: {lt}...'.format(**locals()))
    lt_comps = comps.loc[comps.locus_tag == lt]
    lt_preds = preds.loc[preds.locus_tag == lt]
    lt_parents = choice_lib.pick_n_parents(lt_comps,
                                           lt_preds,
                                           N_FAMILIES)
    old_guides[lt] = choice_lib.choose_n_meas(lt_comps,
                                              OLD_GUIDES_PER_LOCUS)
    new_guides[lt] = choice_lib.choose_n_for_each(lt_parents,
                                                  lt_preds,
                                                  lt_comps,
                                                  NEW_GUIDES_PER_FAMILY)
  import IPython; IPython.embed()
  # TODO(jsh): aggregate and sanity check full sets
