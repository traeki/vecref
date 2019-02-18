#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

from collections import Counter
import logging
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import mapping_lib
import gamma_lib
import training_lib

BASES = 'ACGT'
OLDBASE = pathlib.Path('/home/jsh/gd/proj/lowficrispri/docs/20180626_rebase/')
BSU_TARGETS = OLDBASE / 'data/bsu.NC_000964.targets.all.tsv'
GENES_W_PHENO = OLDBASE / 'output/choose_important_genes.tsv'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

def pick_n_parents(locus_comps, locus_preds, n):
  chosen = Counter()
  fullset = set(locus_preds.original.unique())
  past_picks = set(locus_comps.original.unique())
  unused_picks = fullset - past_picks
  # auto-include past picks but skip pre-filtered
  chosen.update(past_picks.intersection(fullset))
  while sum(chosen.values()) < n:
    if unused_picks:
      pick = random.sample(unused_picks, 1)[0]
      chosen[pick] += 1
      unused_picks.remove(pick)
    else:
      ceiling = chosen.most_common(1)[0][-1] # most common count
      floor = chosen.most_common()[-1][-1] # least common count
      while True:
        # pick elements until we find one with sub-cap count (or we're level)
        boost = random.choice(chosen.most_common())
        if (floor == ceiling) or (boost[1] < ceiling):
          chosen[boost[0]] += 1
          break
  if chosen.most_common(1)[0][-1] > 4:
    locus = locus_comps.locus_tag.unique()[0]
    # 5*10 + 13 > 60, so warn if we see 5+ instances
    template = 'Had to fall back to same family too often: {chosen}'
    logging.warn(template.format(**locals()))
  for ele in chosen.elements():
    yield ele

def choose_n_meas(measured, n):
  chosen = set()
  locus = measured.locus_tag.unique()[0]
  usable = measured.dropna()
  usable = usable.copy()
  if usable.shape[0] < n:
    n_found = usable.shape[0]
    template = 'Only found {n_found}/{n} measured guides for locus {locus}'
    logging.warn(template.format(**locals()))
    return random.sample(set(measured.variant), n)
  # ascribe bins
  usable['bin'] = gamma_lib.bin_relgammas(usable.relgamma,
                                          gamma_lib.relgamma_bins())
  # choose guides for each bin (skipping 0)
  bins = list(range(gamma_lib.NBINS))
  k = n // len(bins)-1
  for b in bins[1:-1]:
    bin_items = usable.loc[usable.bin == b]
    if bin_items.shape[0] <= k:
      # if there aren't more than k items in this bin, grab what we have
      chosen.update(bin_items.variant)
    else:
      # if there are more than k, pick at random
      poss = set(bin_items.variant) - chosen
      chosen.update(random.sample(poss, k))
  # How many more do we need?
  z = (n - len(chosen))
  # Grab up to z preferring non-max efficacy
  leftover = set(usable.variant) - chosen
  toosick = set(usable.loc[usable.bin == bins[0]].variant)
  okset = leftover - toosick
  if len(okset) >= z:
    chosen.update(random.sample(okset, z))
  else:
    chosen.update(okset)
    chosen.update(random.sample(toosick, (z - len(okset))))
  assert len(chosen) == n
  return chosen

def choose_n_for_each(parents, preds, comps, n):
  chosen = set()
  for parent in parents:
    family_preds = preds.loc[preds.original == parent]
    family_comps = comps.loc[comps.original == parent]
    compset = set(family_comps.variant)
    used_mask = family_preds.variant.isin(chosen)
    comp_mask = family_preds.variant.isin(compset)
    family_remaining = family_preds.loc[~used_mask & ~comp_mask]
    if family_remaining.shape[0] < n:
      chosen.update(set(family_remaining.variant))
      delta = n - family_remaining.shape[0]
      template = 'Grabbing {delta} comps for {parent}.'
      logging.warn(template.format(**locals()))
      family_remaining = family_preds.loc[~used_mask & comp_mask]
      compsleft = family_remaining.shape[0]
      if compsleft < delta:
        template = '...but only had {compsleft}/{delta} :-('
        logging.warn(template.format(**locals()))
        chosen.update(set(family_remaining.variant))
    chosen.update(choose_n_by_pred(family_remaining, n))
  return chosen

def choose_n_by_pred(preds, n):
  preds = preds.copy()
  if preds.shape[0] < n:
    locus = preds.iloc[0].locus_tag
    template = 'Fewer than {n} predicted guides for locus {locus}'
    logging.fatal(template.format(**locals()))
  # ascribe bins
  preds['bin'] = gamma_lib.bin_relgammas(preds.y_pred,
                                         gamma_lib.relgamma_bins())
  # choose guides for each bin (skipping 0)
  chosen = set()
  bins = list(range(gamma_lib.NBINS))
  k = n // len(bins)-1
  for b in bins[1:-1]:
    bin_items = preds.loc[preds.bin == b]
    if bin_items.shape[0] <= k:
      # if there aren't more than k items in this bin, grab what we have
      chosen.update(bin_items.variant)
    else:
      # if there are more than k, pick at random
      poss = set(bin_items.variant) - chosen
      chosen.update(random.sample(poss, k))
  # How many more do we need?
  z = (n - len(chosen))
  # Grab up to z preferring non-max efficacy
  leftover = set(preds.variant) - chosen
  toosick = set(preds.loc[preds.bin == bins[0]].variant)
  okset = leftover - toosick
  if len(okset) >= z:
    chosen.update(random.sample(okset, z))
  else:
    chosen.update(okset)
    chosen.update(random.sample(toosick, (z - len(okset))))
  assert len(chosen) == n
  return chosen


def all_single_variants(parents):
  pairs = list()
  for parent in parents:
    assert len(parent) == 20
    children = set()
    for i in range(len(parent)):
      for letter in BASES:
        children.add(parent[:i] + letter + parent[i+1:])
    children.remove(parent)
    for child in children:
      pairs.append((parent, child))
  pairrows = list()
  for original, variant in pairs:
    pairrows.append({'original':original, 'variant':variant})
  return pd.DataFrame(pairrows)

def build_and_filter_pairs():
  parents_frame = pd.read_csv(BSU_TARGETS, sep='\t')
  antisense_rows = parents_frame.loc[parents_frame.transdir=='anti']
  important = set(pd.read_csv(GENES_W_PHENO, sep='\t', header=None)[0])
  important_rows = antisense_rows.loc[antisense_rows.locus_tag.isin(important)]
  dupmask = ~important_rows.target.duplicated(keep=False)
  important_rows = important_rows.loc[dupmask]
  parents = important_rows.target
  origvars = all_single_variants(parents)
  locmap = important_rows[['target', 'locus_tag']].set_index('target').locus_tag
  pammap = important_rows[['target', 'pam']].set_index('target').pam
  origvars['locus_tag'] = origvars.original.map(locmap)
  origvars['pam'] = origvars.original.map(pammap)
  return origvars
