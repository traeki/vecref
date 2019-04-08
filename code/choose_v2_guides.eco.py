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

from predict_all_linear import ECO_PREDFILE
from predict_all_linear import ECO_TARGETS
from predict_all_linear import ECO_LOCI

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
COMPS = UNGD / 'lib2_comps.tsv'
_CODEFILE = pathlib.Path(__file__).name
EXPLORE_FAMILIES = 10
EXPLOIT_FAMILIES = 2
EXPLOIT_GUIDES_PER_LOCUS = 9
EXPLORE_GUIDES_PER_FAMILY = 9
EXPLOITFILE = (UNGD / _CODEFILE).with_suffix('.exploit.tsv')
EXPLOREFILE = (UNGD / _CODEFILE).with_suffix('.explore.tsv')

if __name__ == '__main__':
  logging.info('Reading preds from {ECO_PREDFILE}...'.format(**locals()))
  preds = pd.read_csv(ECO_PREDFILE, sep='\t')
  logging.info('Reading targets from {ECO_TARGETS}...'.format(**locals()))
  all_targets = pd.read_csv(ECO_TARGETS, sep='\t')
  important = set(pd.read_csv(ECO_LOCI, sep='\t', header=None)[0])
  chosen_loci = important
  # loop over locus tags and choose measure
  exploit_guides = dict()
  explore_guides = dict()
  for locus in chosen_loci:
    template = 'Examining options for locus_tag: {locus}...'
    logging.info(template.format(**locals()))
    locus_preds = preds.loc[preds.locus_tag == locus]
    locus_targets = all_targets.loc[all_targets.locus_tag == locus]
    # We don't have "comps", because there *was* no V1 Eco library.
    locus_comps = None
    explore_parents = choice_lib.pick_n_parents(locus_comps,
                                                locus_preds,
                                                locus_targets,
                                                EXPLORE_FAMILIES)
    explore_guides[locus] = choice_lib.choose_n_for_each(explore_parents,
                                                         locus_preds,
                                                         locus_comps,
                                                         EXPLORE_GUIDES_PER_FAMILY)
    exploit_parents = choice_lib.pick_n_parents(locus_comps,
                                                locus_preds,
                                                locus_targets,
                                                EXPLOIT_FAMILIES)
    exploit_parents = list(exploit_parents)
    exploit_candidates = locus_preds.loc[locus_preds.original.isin(exploit_parents)]
    exploit_guides[locus] = choice_lib.choose_n_by_pred(exploit_candidates,
                                                        EXPLOIT_GUIDES_PER_LOCUS)
  allexploit = set()
  for locus in exploit_guides:
    allexploit.update(exploit_guides[locus])
  allexplore = set()
  for locus in explore_guides:
    allexplore.update(explore_guides[locus])
  exploitframe = preds.loc[preds.variant.isin(allexploit)]
  exploitframe.to_csv(EXPLOITFILE, sep='\t', index=False)
  exploreframe = preds.loc[preds.variant.isin(allexplore)]
  exploreframe.to_csv(EXPLOREFILE, sep='\t', index=False)
