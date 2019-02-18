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
  important = set(pd.read_csv(choice_lib.GENES_W_PHENO,
                              sep='\t', header=None)[0])
  # actually loop over locus tags and choose measure
  old_guides = dict()
  new_guides = dict()
  for locus in important:
    template = 'Examining options for locus_tag: {locus}...'
    logging.info(template.format(**locals()))
    locus_comps = comps.loc[comps.locus_tag == locus]
    locus_preds = preds.loc[preds.locus_tag == locus]
    locus_parents = choice_lib.pick_n_parents(locus_comps,
                                              locus_preds,
                                              N_FAMILIES)
    old_guides[locus] = choice_lib.choose_n_meas(locus_comps,
                                                 OLD_GUIDES_PER_LOCUS)
    new_guides[locus] = choice_lib.choose_n_for_each(locus_parents,
                                                     locus_preds,
                                                     locus_comps,
                                                     NEW_GUIDES_PER_FAMILY)
  import IPython; IPython.embed()
  # TODO(jsh): aggregate and sanity check full sets
