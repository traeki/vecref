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

def pick_n_parents(lt_comps, lt_preds, n):
  chosen = Counter()
  possible_picks = lt_preds.original.unique()
  past_picks = lt_comps.original.unique()
  unused_picks = set(possible_picks) - set(past_picks)
  # auto-include past picks but skip pre-filtered (i.e. not in possible_picks)
  chosen.update(set(past_picks) - set(possible_picks))
  while len(list(chosen.elements())) < n:
    if unused_picks:
      pick = random.choice(unused_picks)
      unused_picks[pick] += 1
      unused_picks -= pick
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
    # 5*10 + 13 > 60, so bail if we see 5+ instances
    template = 'Had to fall back to same family too often: {lt}'
    logging.fatal(template.format(**locals()))
    sys.exit(13)
  for ele in chosen.elements():
    yield ele

def choose_n_meas(measured, n):
  chosen = set()
  # TODO(jsh): Actually choose n guides (divided over loci) with good coverage.
  return chosen

def choose_n_for_each(parents, preds, comps, n):
  chosen = set()
  # TODO(jsh): Actually choose n guides for each parent, no comps.
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
