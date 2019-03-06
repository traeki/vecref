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

from predict_all_linear import PREDFILE as linear_preds

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
COMPS = UNGD / 'lib2_comps.tsv'
_CODEFILE = pathlib.Path(__file__).name
N_LOCI = 300
N_FAMILIES = 10
OLD_GUIDES_PER_LOCUS = 9
NEW_GUIDES_PER_FAMILY = 9
OLDGUIDEFILE = (UNGD / _CODEFILE).with_suffix('.old.guides.tsv')
NEWGUIDEFILE = (UNGD / _CODEFILE).with_suffix('.new.guides.tsv')

if __name__ == '__main__':
  logging.info('Reading preds from {linear_preds}...'.format(**locals()))
  preds = pd.read_csv(linear_preds, sep='\t')
  logging.info('Reading comps from {COMPS}...'.format(**locals()))
  comps = pd.read_csv(COMPS, sep='\t')
  var_rg = mapping_lib.get_mapping('variant', 'unfiltered_relgamma', UNGD)
  var_rg.rename(columns={'unfiltered_relgamma':'relgamma'}, inplace=True)
  comps['relgamma'] = comps.variant.map(var_rg.relgamma)
  important = set(pd.read_csv(choice_lib.GENES_W_PHENO,
                              sep='\t', header=None)[0])
  essmap = mapping_lib.get_mapping('locus_tag', 'bmk_ess', UNGD)
  kinda = (important - set(essmap.loc[essmap.bmk_ess == True].index))
  veryimp = important - kinda
  fill_size = N_LOCI - len(veryimp)
  # grab mean parent gamma for all such loci
  var_orig = mapping_lib.get_mapping('variant', 'original', UNGD)
  var_loc = mapping_lib.get_mapping('variant', 'locus_tag', UNGD)
  locsub = var_loc.loc[var_loc.locus_tag.isin(kinda)]
  var_orig.reset_index(inplace=True)
  origs = var_orig.loc[var_orig.variant == var_orig.original]
  bothsub = set(origs.loc[origs.variant.isin(locsub.index)].variant)
  pgframe = origs.loc[origs.variant.isin(locsub.index)].copy()
  pgframe['locus_tag'] = pgframe.variant.map(var_loc.locus_tag)
  var_dir = mapping_lib.get_mapping('variant', 'rel_dir', UNGD)
  pgframe['rel_dir'] = pgframe.variant.map(var_dir.rel_dir)
  pgframe = pgframe.loc[pgframe.rel_dir == 'anti'].copy()
  var_gamma = mapping_lib.get_mapping('variant', 'gamma', UNGD)
  pgframe['gamma'] = pgframe.variant.map(var_gamma.gamma)
  locus_mpg = pgframe.groupby('locus_tag').mean().gamma
  # Drop down to the right number
  fill_loci = set(locus_mpg.sort_values().head(fill_size).index)
  chosen_loci = veryimp.union(fill_loci)
  # loop over locus tags and choose measure
  old_guides = dict()
  new_guides = dict()
  for locus in chosen_loci:
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
  allold = set()
  for locus in old_guides:
    allold.update(old_guides[locus])
  allnew = set()
  for locus in new_guides:
    allnew.update(new_guides[locus])
  oldframe = comps.loc[comps.variant.isin(allold)]
  oldframe.to_csv(OLDGUIDEFILE, sep='\t', index=False)
  newframe = preds.loc[preds.variant.isin(allnew)]
  newframe.to_csv(NEWGUIDEFILE, sep='\t', index=False)
