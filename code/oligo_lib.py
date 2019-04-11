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

BSU_EXPLOIT = UNGD / 'choose_v2_guides.bsu.exploit.tsv'
BSU_EXPLORE = UNGD / 'choose_v2_guides.bsu.explore.tsv'
ECO_EXPLOIT = UNGD / 'choose_v2_guides.eco.exploit.tsv'
ECO_EXPLORE = UNGD / 'choose_v2_guides.eco.explore.tsv'
DFRA_FILE =  UNGD / 'bsu.dfrA.all.tsv'
MURAA_FILE = UNGD / 'bsu.murAA.all.tsv'
FOLA_FILE =  UNGD / 'eco.folA.all.tsv'
MURA_FILE =  UNGD / 'eco.murA.all.tsv'

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


def oligos_from(filename, adaptid, ncopies):
  guides = pd.read_csv(filename, sep='\t')
  variants = build_oligos(guides.variant, adaptid, ncopies=ncopies)
  parents = build_oligos(guides.original.unique(), adaptid, ncopies=ncopies)
  n_total = (len(guides.variant) + len(guides.original.unique()))
  logging.info('cooked up {n_total} oligos'.format(**locals()))
  n_parents = len(guides.original.unique())
  logging.info('...{n_parents} of them parents'.format(**locals()))
  return (variants + parents)

if __name__ == '__main__':

  con_copies = 9
  bsu_exploit_copies = 0
  bsu_explore_copies = 0
  eco_exploit_copies = 4
  eco_explore_copies = 2
  dfra_copies = 4
  muraa_copies = 4
  fola_copies = 4
  mura_copies = 4

  con_adaptid = 0
  bsu_exploit_adaptid = 1
  bsu_explore_adaptid = 2
  eco_exploit_adaptid = 3
  eco_explore_adaptid = 4
  dfra_adaptid = 5
  muraa_adaptid = 6
  fola_adaptid = 7
  mura_adaptid = 8

  cmap = mapping_lib.get_mapping('variant', 'control', UNGD)
  controls = list(cmap.loc[cmap.control].index)
  colis = build_oligos(controls,
                       con_adaptid,
                       ncopies=con_copies)

  with open(OLIGOFILE, 'w') as outfile:
    allolis = list()
    allolis.extend(colis)
    allolis.extend(
        oligos_from(BSU_EXPLOIT, bsu_exploit_adaptid, bsu_exploit_copies))
    allolis.extend(
        oligos_from(BSU_EXPLORE, bsu_explore_adaptid, bsu_explore_copies))
    allolis.extend(
        oligos_from(ECO_EXPLOIT, eco_exploit_adaptid, eco_exploit_copies))
    allolis.extend(
        oligos_from(ECO_EXPLORE, eco_explore_adaptid, eco_explore_copies))
    allolis.extend(
        oligos_from(DFRA_FILE, dfra_adaptid, dfra_copies))
    allolis.extend(
        oligos_from(MURAA_FILE, muraa_adaptid, muraa_copies))
    allolis.extend(
        oligos_from(FOLA_FILE, fola_adaptid, fola_copies))
    allolis.extend(
        oligos_from(MURA_FILE, mura_adaptid, mura_copies))
    outfile.write('\n'.join(allolis))
