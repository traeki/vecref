#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

import numpy as np
import pandas as pd

from Bio import SeqIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
np.set_printoptions(precision=4, suppress=True)

_MAPPING_TEMPLATE = 'mapping.{0}.{1}{2}.tsv'
def make_mapping(frame, a, b, datadir, dose=None):
  infix = ''
  if dose:
    infix = '.' + dose
  mapping = [a, b]
  mapview = frame[mapping].copy()
  mapview.drop_duplicates(inplace=True)
  outfile = datadir / _MAPPING_TEMPLATE.format(a, b, infix)
  mapview.to_csv(outfile, sep='\t', index=False)
def get_mapping(a, b, datadir, dose=None):
  infix = ''
  if dose:
    infix = '.' + dose
  tsv_file = datadir / _MAPPING_TEMPLATE.format(a, b, infix)
  frame = pd.read_csv(tsv_file, sep='\t', header=0, index_col=a)
  return frame

_COUNTDATA_FILENAME = 'sample.countdata.tsv'
def get_countdata(datadir):
  return pd.read_csv(datadir / _COUNTDATA_FILENAME, sep='\t', header=0)

_OD_DATA_FILENAME = 'od_data.tsv'
def get_od_data(datadir):
  return pd.read_csv(datadir / _OD_DATA_FILENAME, sep='\t', header=0)

def mappings_from_target_file(target_file, datadir):
  target_data = pd.read_csv(
      target_file,
      sep='\t',
      names=['locus_tag', 'offset', 'variant', 'pam',
        'chrom', 'start', 'end', 'abs_dir', 'rel_dir',
        'weakness', 'specificity'])
  mappings = list()
  mappings.append(['variant', 'pam'])
  mappings.append(['variant', 'specificity'])
  mappings.append(['variant', 'weakness'])
  mappings.append(['variant', 'locus_tag'])
  mappings.append(['variant', 'offset'])
  mappings.append(['variant', 'abs_dir'])
  mappings.append(['variant', 'rel_dir'])
  for mapping in mappings:
    make_mapping(target_data, *mapping, datadir)

def adapt_orig_map(orig_map_file, datadir):
  orig_data = pd.read_csv(orig_map_file, sep='\t', header=0)
  # TODO(jsh): fix this when updating adaptation, but for now...
  # TODO(jsh): ...discard pairs where "original" has non-zero weakness
  weakmap = get_mapping('variant', 'weakness', datadir)
  check = pd.merge(orig_data, weakmap,
                   how='left',
                   left_on='original', right_index=True)
  spurious = (check.weakness != 0) & (check.weakness.notna())
  orig_data = orig_data.loc[~spurious]
  make_mapping(orig_data, *(orig_data.columns), datadir)

def adapt_gene_map(gene_map_file, datadir):
  names = ['locus_tag', 'gene_name']
  gene_data = pd.read_csv(gene_map_file, sep='\t', names=names)
  make_mapping(gene_data, *(gene_data.columns), datadir)

def read_countfiles(countfile_dir, pattern, datadir):
  """Read the count files into columns."""
  def get_sample(countfile):
    sample = countfile.name.split('_')[0]
    frame = pd.read_csv(countfile, sep='\t', names=['variant', 'raw'])
    frame.raw = frame.raw.astype('int')
    if sample.startswith('t'):
      alts = list()
      for tube in ['a', 'b', 'c']:
        alias = tube + sample[1:]
        aliased = frame.copy()
        aliased['sample'] = alias
        alts.append(aliased)
      return pd.concat(alts, axis='index')
    else:
      frame['sample'] = sample
      return frame
  countfiles = countfile_dir.glob(pattern)
  samples = [get_sample(countfile) for countfile in countfiles]
  countdata = pd.concat(samples, axis='index')
  countdata.reset_index(drop=True, inplace=True)
  outfile = datadir / _COUNTDATA_FILENAME
  countdata = countdata[['variant', 'sample', 'raw']]
  countdata.to_csv(outfile, sep='\t', index=False)

def map_locus_tag_to_len(genome, datadir):
  mapping = dict()
  g = SeqIO.parse(genome, 'genbank')
  for chrom in g:
    for feature in chrom.features:
      if feature.type == 'gene':
        locus = feature.qualifiers['locus_tag'][0]
        mapping[locus] = abs(feature.location.end - feature.location.start)
  frame = pd.DataFrame.from_dict(mapping, orient='index', columns=['locus_len'])
  frame.index.name = 'locus_tag'
  frame.reset_index(inplace=True)
  make_mapping(frame, 'locus_tag', 'locus_len', datadir)

def process_bmk_spreadsheet(jmpbmk_annos, datadir):
  bmk_data = pd.read_excel(jmpbmk_annos)
  bmk_data.rename(columns={
    bmk_data.columns[9]: 'lowfit',
    bmk_data.columns[11]: 'bmk_ess'}, inplace=True)
  bmk_data = bmk_data[[
    'locus_tag', 'gene_name', 'lowfit',
    'bmk_ess', 'small_transformation']]
  bmk_data.lowfit = (bmk_data.lowfit == 1)
  bmk_data['small'] = (bmk_data.small_transformation == 'yes')
  bmk_data.drop('small_transformation', 1, inplace=True)
  bmk_data['bmk_sick'] = bmk_data.small | bmk_data.lowfit
  bmk_data.bmk_ess = (bmk_data.bmk_ess == 'essential')
  mappings = list()
  mappings.append(['locus_tag', 'bmk_ess'])
  mappings.append(['locus_tag', 'bmk_sick'])
  for mapping in mappings:
    make_mapping(bmk_data, *mapping, datadir)

def map_variant_to_oneoff(orig_map_frame, datadir):
  def is_oneoff(row):
    variant, original = row['variant'], row['original']
    for i in range(len(variant)):
      if variant[i] != original[i]:
        fixone = variant[:i] + original[i] + variant[i+1:]
        # that's the one mismatch IFF fixing it works, so we're done either way
        return fixone == original
    return False
  orig_map_frame['is_oneoff'] = orig_map_frame.apply(is_oneoff, axis=1)
  make_mapping(orig_map_frame, 'variant', 'is_oneoff', datadir)

_SAMPLE_TAG_FILE = 'sample.tags.tsv'
def tags_from_sample(sample):
  sid = sample[2:] + sample[:1]
  tp = int(sample[1])
  dose = 'sober'
  if sid[2] != 'a':
    if sid[:2] == 'd2':
      dose = 'low'
    if sid[:2] == 'd3':
      dose = 'high'
  return pd.Series({'sid': sid, 'tp': tp, 'dose': dose})
def make_sample_tags(datadir):
  samples = get_countdata(datadir)[['sample']].drop_duplicates()
  sampletags = samples['sample'].apply(tags_from_sample)
  samplemap = pd.merge(samples, sampletags,
                       left_index=True, right_index=True)
  samplemap.to_csv(datadir / _SAMPLE_TAG_FILE, sep='\t', index=False)
def get_sample_tags(datadir):
  return pd.read_csv(datadir / _SAMPLE_TAG_FILE, sep='\t', header=0)


def adapt_od_data(od_data_file, datadir):
  # start with : time / od / day / tube / sample
  # need: sid / sample / time / od
  od_data = pd.read_csv(od_data_file, sep='\t')
  tags = od_data['sample'].apply(tags_from_sample)
  outframe = pd.merge(tags[['sid']], od_data[['sample', 'time', 'od']],
                      left_index=True, right_index=True)
  outfile = datadir / _OD_DATA_FILENAME
  outframe.to_csv(outfile, sep='\t', index=False)

def make_variant_controltag_map(datadir):
  data = pd.read_csv(datadir / _COUNTDATA_FILENAME, sep='\t', header=0)
  varloc = get_mapping('variant', 'locus_tag', datadir).reset_index()
  joined = pd.merge(data[['variant']], varloc, how='left', on='variant')
  joined.drop_duplicates(inplace=True)
  joined['control'] = joined.locus_tag.isna()
  make_mapping(joined, 'variant', 'control', datadir)
