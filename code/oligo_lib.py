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

from Bio.Seq import Seq

import mapping_lib

from choose_v2_guides import OLDGUIDEFILE
from choose_v2_guides import NEWGUIDEFILE

ADAPTERS = list()
ADAPTERS.append(('ATTTTGCCCCTGGTTCTT', 'CCAGTTCATTTCTTAGGG'))
ADAPTERS.append(('TCACAACTACACCAGAAG', 'GCAACACTTTGACGAAGA'))
ADAPTERS.append(('CTGTGTAATCTCCGACAC', 'GCCTTTGCATGTTGTGGA'))
ADAPTERS.append(('GCGTGTTTGAATTCCACT', 'AAATTTCCTCGTCGGCTC'))
ADAPTERS.append(('AGGATCTCTAGCCTCAAA', 'GATGAAGCATCGTAACTG'))
ADAPTERS.append(('AACTGCGATCGCTAATGT', 'GTTCTCCAGTGCCTTATT'))
ADAPTERS.append(('CTTCTACCGAACATACAG', 'TCGCGTTATGCTGTATGT'))
ADAPTERS.append(('CGTAGCGTTTTGTACACG', 'CACGCTCAAAAAGCGTAC'))
ADAPTERS.append(('GGCATGCTGCAATAACCT', 'AAACCGGTGAGCTGGAAT'))
ADAPTERS.append(('CCGGTAACTATTCTAGCC', 'GCAGCCCGAATACTTTCA'))
ADAPTERS.append(('ATACGTTAACCCGTATGG', 'TCTAGCCTACAATTCACG'))
ADAPTERS.append(('CCCTAAGCATTCGCGAAA', 'ATTGTTAGCGCCACAATC'))
ADAPTERS.append(('CATTTCTAGCCCTTCGAG', 'AACTCAACTTGGCAGGAA'))
ADAPTERS.append(('GGATTGGACAAGCTAGTT', 'TAATACAAGGCCGCGTTG'))
ADAPTERS.append(('GGGGGAAAAGGATCGATT', 'TCACGTGTAAACGATGCG'))
ADAPTERS.append(('ACCCCAGTTGTGAATATC', 'TCGGATGACCCTAGAAAG'))
ADAPTERS.append(('TATGAACCACTAAGGCGT', 'CGTAAAGTCTGCTGGTGA'))
ADAPTERS.append(('GCTGGTGTTATGGTGGAA', 'AAGTCGTTTTTGGACCGC'))
ADAPTERS.append(('GAGCACACACAAGAATGA', 'TAGCGACAACGTCACAAC'))
ADAPTERS.append(('CACACTATGTCATCCGCA', 'CCTTTAGCAAGCAAAGCC'))
ADAPTERS.append(('TAGTCAGAGAGTCGAGAG', 'GGTTTGGAGCCATTAGTT'))
ADAPTERS.append(('GTAGACAAATGCTTGGAC', 'TCCCAGTCTAGTATGAGG'))
ADAPTERS.append(('AAGGCCCAAGTCGCTTTT', 'CCGCCTTTTTCAAGTGAT'))
ADAPTERS.append(('TCGATCCGGGAGTATACA', 'CCTACAAGAGTTCGACAC'))
ADAPTERS.append(('ACATCCTGGTTACTTGGC', 'TGCTCTCGATCATAGCCT'))
ADAPTERS.append(('GAAATATTAAGGGCGGCT', 'TTCGGTCAATAGAGTCGG'))
ADAPTERS.append(('AATATGTCCCGTTCCTAC', 'CACTCTGTTCGGAAAACT'))
ADAPTERS.append(('TTGGAAAGCAGTACTGCA', 'GATGTGTAAGTGGGCCTA'))
ADAPTERS.append(('AGAGATCCGGTGTCAAAG', 'CCCAGTTCAATCGCTCAA'))
ADAPTERS.append(('CGACTGGACGGATTTTGA', 'CGTTCCTCCGCCTTATTA'))
ADAPTERS.append(('TGCGTAATGCATGTGATC', 'TGCTATTAAGATCCTCCC'))
ADAPTERS.append(('ATCATCAATCACTCCCCG', 'TCGATCTTGAGAAGCAGT'))
ADAPTERS.append(('GTCTATTGAAGTACCTGC', 'TATCTTGGAGGAGCATCG'))
ADAPTERS.append(('TGGCATTGCTTCGTCAAG', 'CATGACGACTTCACCCAT'))
ADAPTERS.append(('CCAACATGACGTTCTGTC', 'GAACGTCGAGTAAATGTC'))
ADAPTERS.append(('TTTACGGTCCACCATTTG', 'AACGTGAGGATTAGCGCT'))
ADAPTERS.append(('GACGTGGACTTGGACAAA', 'GAACTGTGCGATTTGCAG'))
ADAPTERS.append(('TGTTTGACAATCTCGGGC', 'GCACATAAGTACCACTCC'))
ADAPTERS.append(('CAAGCGCTAAGCACGAAA', 'AGTGATATCTACCGCGTG'))
ADAPTERS.append(('TTATTAGTTTGTGCCCGC', 'AAAAGCATACAGGCACCA'))
ADAPTERS.append(('CTCGGGATAGTTATACTC', 'CTTAGTGTAGATTTGGGC'))
BSA_5P = 'GGTCTCT'
HANG_5P = 'ATGT'
HANG_3P = 'GTTT'
BSA_3P = 'AGAGACC'
UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
_CODEFILE = pathlib.Path(__file__).name
OLIGOFILE = (UNGD / _CODEFILE).with_suffix('.oligos')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

def build_oligos(guides, adapterid, *, bothdirs=True, ncopies=1):
  oligos = list()
  a5p, a3p = ADAPTERS[adapterid]
  for g in guides:
    # TODO(jsh): figure out how to actually get right label set without hack
    allvars = dict()
    allvars.update(locals())
    allvars.update(globals())
    # TODO(jsh): figure out how to actually get right label set without hack
    template = '{a5p}{BSA_5P}{HANG_5P}{g}{HANG_3P}{BSA_3P}{a3p}'
    dna = template.format(**allvars)
    dna = Seq(dna)
    oligos.append(str(dna))
    if bothdirs:
      oligos.append(str(dna.reverse_complement()))
  collated = list()
  for i in range(ncopies):
    collated.extend(oligos)
  return collated

if __name__ == '__main__':
  con_copies = 3
  old_copies = 3
  new_copies = 3
  con_adaptid = 0
  old_adaptid = 1
  new_adaptid = 2
  cmap = mapping_lib.get_mapping('variant', 'control', UNGD)
  controls = list(cmap.loc[cmap.control].index)
  colis = build_oligos(controls,
                       con_adaptid,
                       ncopies=con_copies)
  oldguides = pd.read_csv(OLDGUIDEFILE, sep='\t')
  oldvariants = build_oligos(oldguides.variant,
                             old_adaptid,
                             ncopies=old_copies)
  oldparents = build_oligos(oldguides.original.unique(),
                            old_adaptid,
                            ncopies=old_copies)
  nold = len(oldguides.variant) + len(oldguides.original.unique())
  logging.info('{nold} old oligos'.format(**locals()))
  noldps = len(oldguides.original.unique())
  logging.info('...{noldps} of them parents'.format(**locals()))
  newguides = pd.read_csv(NEWGUIDEFILE, sep='\t')
  newvariants = build_oligos(newguides.variant,
                             new_adaptid,
                             ncopies=new_copies)
  newparents = build_oligos(newguides.original.unique(),
                            new_adaptid,
                            ncopies=new_copies)
  nnew = len(newguides.variant) + len(newguides.original.unique())
  logging.info('{nnew} new oligos'.format(**locals()))
  nnewps = len(newguides.original.unique())
  logging.info('...{nnewps} of them parents'.format(**locals()))
  with open(OLIGOFILE, 'w') as outfile:
    allolis = list()
    allolis.extend(colis)
    allolis.extend(oldvariants)
    allolis.extend(oldparents)
    allolis.extend(newvariants)
    allolis.extend(newparents)
    outfile.write('\n'.join(allolis))
