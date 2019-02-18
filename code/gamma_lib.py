#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib
import sys

import numpy as np
import pandas as pd

import mapping_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

_NORMAL_SIZE = float(50 * 1000 * 1000)
_THRESHOLD = 50
_PSEUDO = 1
_Z_THRESHOLD = 12
BIN_MIN = -0.9
BIN_MAX = -0.1
NBINS = 5

def _namespan_func(k):
  def namespan(id_pair):
    (sid, tp) = id_pair
    front, back = tp-k, tp
    return '{sid}{front}{back}'.format(**vars())
  return namespan

def _diff_by_tp(group, field, *, k=1, raw_threshold=None):
  tall = group[field]
  wide = tall.unstack().copy()
  wide.sort_index(axis=1, inplace=True)
  wide -= wide.shift(k, axis=1)
  wide.drop(range(k), axis=1, inplace=True)
  if raw_threshold is not None:
    mask = group['raw'] > raw_threshold
    mask = mask.unstack()
    mask.sort_index(axis=1, inplace=True)
    mask |= mask.shift(k, axis=1)
    wide = wide.where(mask)
  return wide.stack(dropna=False)

def _drop_drugged(data):
  return data.drop(data.loc[data.dose != 'sober'].index)

_ROUGH_GAMMA_FILE = 'rough.gammas.by.variant.tsv'
def compute_rough_gammas(datadir):
  data = mapping_lib.get_countdata(datadir)
  varcon = mapping_lib.get_mapping('variant', 'control', datadir)
  data = data.merge(varcon.reset_index(), how='left', on='variant')
  data = data.merge(mapping_lib.get_sample_tags(datadir),
                    how='left', on='sample')
  data = _drop_drugged(data)
  data.drop('sample', axis=1, inplace=True)
  def normalize(counts):
    return counts * _NORMAL_SIZE / counts.sum()
  data['norm'] = data.groupby(['sid', 'tp']).raw.transform(normalize)
  data['log'] = np.log2(data.norm.clip(_PSEUDO))
  data.set_index(['variant', 'sid', 'tp'], inplace=True)
  grouper = data.groupby(['sid'], group_keys=False)
  relevant = list()
  for i in range(1, 4):
    namespan = _namespan_func(i)
    diff = grouper.apply(_diff_by_tp, 'log', k=i, raw_threshold=_THRESHOLD)
    diffcenters = diff.loc[data.control].unstack(level=[-2,-1]).median()
    dg = diff.unstack(level=[-2,-1]).subtract(diffcenters, axis='columns')
    dg.columns = dg.columns.map(namespan)
    relevant.append(dg)
  X = pd.concat(relevant, axis=1)
  X.to_csv(datadir / _ROUGH_GAMMA_FILE, sep='\t')
def get_rough_gammas(datadir):
  X = pd.read_csv(datadir / _ROUGH_GAMMA_FILE,
                  sep='\t', header=0, index_col='variant')
  return X

def rebase(A, D):
  A = np.ma.masked_invalid(A)
  U_, s_, Vt_ = np.linalg.svd(D, full_matrices=True)
  rank_ = (~np.isclose(s_, 0)).sum()
  basis_ = U_[:, :rank_]
  unmasked = np.ma.dot(A, np.ma.dot(basis_, basis_.T))
  masked = unmasked
  masked.mask = A.mask
  return masked


def dca_smooth_gammas(unsmoothed_gammas):
  X = unsmoothed_gammas
  taggers = list()
  taggers.append(('span01', lambda s: (s[-2:] == '01')))
  taggers.append(('span02', lambda s: (s[-2:] == '02')))
  taggers.append(('span03', lambda s: (s[-2:] == '03')))
  taggers.append(('span12', lambda s: (s[-2:] == '12')))
  taggers.append(('span13', lambda s: (s[-2:] == '13')))
  taggers.append(('span23', lambda s: (s[-2:] == '23')))
  taggers.append(('none', lambda s: (s[2] == 'a' or s[:2] == 'd1')))
  taggers.append(('low', lambda s: (s[2] != 'a' and s[:2] == 'd2')))
  taggers.append(('high', lambda s: (s[2] != 'a' and s[:2] == 'd3')))
  taggers.append(('a', lambda s: (s[2] == 'a')))
  taggers.append(('b', lambda s: (s[2] == 'b')))
  taggers.append(('c', lambda s: (s[2] == 'c')))
  taggers.append(('d1', lambda s: (s[:2] == 'd1')))
  taggers.append(('d2', lambda s: (s[:2] == 'd2')))
  taggers.append(('d3', lambda s: (s[:2] == 'd3')))
  D = np.asarray([[tagger(col) for (_, tagger) in taggers] for col in X])
  XDDt = pd.DataFrame(rebase(X, D), index=X.index, columns=X.columns)
  XDDt.columns.set_names('spid', inplace=True)
  return XDDt

_DCA_SMOOTHED_GAMMA_FILE = 'dca.smoothed.gammas.tsv'
def compute_dca_smooth_gammas(datadir):
  X = get_rough_gammas(datadir)
  XDDt = dca_smooth_gammas(X)
  XDDt.to_csv(datadir / _DCA_SMOOTHED_GAMMA_FILE, sep='\t')
def get_dca_smooth_gammas(datadir):
  XDDt = pd.read_csv(datadir / _DCA_SMOOTHED_GAMMA_FILE,
                     sep='\t', header=0, index_col='variant')
  return XDDt

def normalize_gammas(countdata, sampletags, XDDt, od_data):
  def g_fit(od_group):
    g, _ = np.polyfit(od_group.time, np.log2(od_group.od), 1)
    return g
  g_map = [[sid, g_fit(group)] for sid, group in od_data.groupby('sid')]
  g_map = pd.DataFrame(g_map, columns=['sid', 'g_fit'])
  data = pd.merge(countdata, sampletags, how='left', on='sample')
  data = _drop_drugged(data)
  data = pd.merge(data, od_data[['sample', 'time']], how='left', on='sample')
  data.drop('sample', axis=1, inplace=True)
  data.set_index(['variant', 'sid', 'tp'], inplace=True)
  grouper = data.groupby(['sid'], group_keys=False)
  relevant = list()
  for i in range(1, 4):
    namespan = _namespan_func(i)
    diff = grouper.apply(_diff_by_tp, 'time', k=i)
    grid = diff.unstack(level=[-2,-1])
    grid.columns = grid.columns.map(namespan)
    relevant.append(grid)
  t_map = pd.concat(relevant, axis=1).iloc[0]
  t_map.name = 'delta_t'
  gt_map = pd.DataFrame(t_map)
  gt_map.index.name = 'spid'
  gt_map['sid'] = gt_map.index.map(lambda x: x[:3])
  gt_map = pd.merge(gt_map.reset_index(), g_map, on='sid', how='left')
  gt_map.drop(['sid'], axis=1, inplace=True)
  gt_map.set_index('spid', inplace=True)
  flatdf = XDDt / (gt_map.g_fit * gt_map.delta_t)
  parts = [(x[:3], x[3:]) for x in flatdf.columns]
  flatdf.columns = pd.MultiIndex.from_tuples(parts, names=['sid', 'span'])
  flatdf.sort_index(axis=1, inplace=True)
  return flatdf

_NORMED_DCA_GAMMA_FILE = 'normed.dca.gammas.tsv'
def compute_normed_gammas(datadir):
  sampletags = mapping_lib.get_sample_tags(datadir)
  countdata = mapping_lib.get_countdata(datadir)
  od_data = mapping_lib.get_od_data(datadir)
  XDDt = get_dca_smooth_gammas(datadir)
  flatdf = normalize_gammas(countdata, sampletags, XDDt, od_data)
  flatdf.to_csv(datadir / _NORMED_DCA_GAMMA_FILE, sep='\t')
def get_normed_gammas(datadir):
  gammas = pd.read_csv(datadir / _NORMED_DCA_GAMMA_FILE,
                       sep='\t', header=[0,1], index_col=0)
  return gammas

def map_variant_to_mean_full_gamma(datadir):
  allgammas = get_normed_gammas(datadir)
  allgammas = allgammas.stack(level=1, dropna=False).reset_index()
  gammas = allgammas.loc[allgammas.span == '03']
  gammas = gammas.drop('span', axis=1)
  gammas.set_index('variant', inplace=True)
  gammas = pd.DataFrame(gammas.mean(axis=1), columns=['gamma'])
  gammas.reset_index(inplace=True)
  mapping_lib.make_mapping(gammas, 'variant', 'gamma', datadir)

_CHILD_GAMMA_FILE = 'child.gammas.tsv'
_PARENT_GAMMA_FILE = 'parent.gammas.tsv'
def derive_child_parent_gammas(datadir):
  var_orig = mapping_lib.get_mapping('variant', 'original', datadir)
  var_orig.reset_index(inplace=True)
  allgammas = get_normed_gammas(datadir)
  parent_gammas = allgammas.loc[var_orig.original]
  child_gammas = allgammas.loc[var_orig.variant]
  parent_gammas.index = child_gammas.index
  child_gammas.to_csv(datadir / _CHILD_GAMMA_FILE, sep='\t')
  parent_gammas.to_csv(datadir / _PARENT_GAMMA_FILE, sep='\t')
def get_child_gammas(datadir):
  gammas = pd.read_csv(datadir / _CHILD_GAMMA_FILE,
                       sep='\t', header=[0,1], index_col=0)
  return gammas
def get_parent_gammas(datadir):
  gammas = pd.read_csv(datadir / _PARENT_GAMMA_FILE,
                       sep='\t', header=[0,1], index_col=0)
  return gammas

def map_variant_to_mean_full_relative_gamma(datadir):
  child_gammas = get_child_gammas(datadir)
  parent_gammas = get_parent_gammas(datadir)
  varcon = mapping_lib.get_mapping('variant', 'control', datadir)
  vargamma = mapping_lib.get_mapping('variant', 'gamma', datadir)
  conmask = vargamma.index.intersection(varcon.loc[varcon.control].index)
  congamma = vargamma.loc[conmask]
  sigma = congamma.std().gamma
  z = -sigma # easier to read
  unfiltered = (child_gammas / parent_gammas) - 1
  geodelt_gammas = unfiltered.where(parent_gammas < (_Z_THRESHOLD * z))
  geodelt_gammas = geodelt_gammas.stack(level=1, dropna=False).reset_index()
  relgammas = geodelt_gammas.loc[geodelt_gammas.span == '03']
  relgammas = relgammas.drop('span', axis=1)
  relgammas.set_index('variant', inplace=True)
  relgammas = pd.DataFrame(relgammas.mean(axis=1), columns=['relgamma'])
  relgammas.reset_index(inplace=True)
  mapping_lib.make_mapping(relgammas, 'variant', 'relgamma', datadir)

def relgamma_bins():
  return np.linspace(BIN_MIN, BIN_MAX, NBINS-1)

def bin_relgammas(relgammas, bins):
  rgbins = np.digitize(relgammas, bins)
  return rgbins

def map_variant_to_bin(datadir):
  varrg = mapping_lib.get_mapping('variant', 'relgamma', datadir)
  bins = relgamma_bins()
  rgbin = bin_relgammas(varrg.relgamma.values, bins)
  rgbin = pd.DataFrame(rgbin.T, index=varrg.index, columns=['rgbin']).reset_index()
  mapping_lib.make_mapping(rgbin, 'variant', 'rgbin', datadir)

def unfiltered_mean_relgammas(datadir):
  child_gammas = get_child_gammas(datadir)
  parent_gammas = get_parent_gammas(datadir)
  geodelt_gammas = (child_gammas / parent_gammas) - 1
  geodelt_gammas = geodelt_gammas.stack(level=1, dropna=False).reset_index()
  relgammas = geodelt_gammas.loc[geodelt_gammas.span == '03']
  relgammas = relgammas.drop('span', axis=1)
  relgammas.set_index('variant', inplace=True)
  relgammas = pd.DataFrame(relgammas.mean(axis=1), columns=['relgamma'])
  return relgammas

def weight_bins(binlabels):
  bounds = np.concatenate([[-1], relgamma_bins(), [0]])
  binwidths = np.diff(bounds)
  bincounts = binlabels.value_counts()
  binratios = bincounts / binlabels.count()
  binweights = dict(binwidths / binratios)
  return binweights
