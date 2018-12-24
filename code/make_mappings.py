#!/usr/bin/env python
# Author: John Hawkins (jsh) [really@gmail.com]

import logging
import pathlib

import mapping_lib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

UNGD = pathlib.Path('/home/jsh/ungd/proj/vecref')
DIR_PREFIX = pathlib.Path(__file__).parents[1]
STATICDIR = DIR_PREFIX / 'static'

# TODO(jsh): move count_guides code and fastq over to the new structure

# FUNDAMENTAL INPUTS
COUNT_GLOB = '??d?_*L00*.fastq.counts'
OD_DATA = STATICDIR / '20171205.od.v.time.tsv'
ALL_OLIGOS = STATICDIR / 'hawk1234.oligos'
TARGET_FILE = STATICDIR / 'lib234.targets.joined.tsv'
GENOME_FILE = STATICDIR / 'bsu.NC_000964.gb'
JMPBMK_ANNOS = STATICDIR / 'jmpbmk.annos.xlsx'
# NOTE(jsh):
  # orig_map.tsv is actually derived by comparing oligos directly to genome:
  # genome ->
  # oligos ->
  # ../lowficrispri/docs/20171205_lib2_ind/code/20180619.ipython.neworigmap.py
  # -> orig_map.tsv
  # TODO(jsh): add a function to recreate this file directly from genome/oligos
  # TODO(jsh): MAKE CAREFUL NOTE OF TODO IN MAPPING_LIB.PY!!!
  # TODO(jsh): ...then remove this "static" file
ORIG_MAP = STATICDIR / 'orig_map.tsv'
# NOTE(jsh):
  # locus.gene.map was manually created from genbank via ipython
  # TODO(jsh): add a function to recreate this file from genbank consistently
  # TODO(jsh): ...then remove this "static" file
GENE_MAP = STATICDIR / 'bsu.NC_000964.gb.locus.gene.map.tsv'

# variant -> pam
# variant -> specificity
# variant -> weakness
# variant -> locus_tag
# variant -> offset
# variant -> abs_dir
# variant -> rel_dir
mapping_lib.mappings_from_target_file(TARGET_FILE, UNGD)

# variant -> original
# TODO(jsh): fix this (see above)
mapping_lib.adapt_orig_map(ORIG_MAP, UNGD)

# locus_tag -> gene
# TODO(jsh): fix this (see above)
mapping_lib.adapt_gene_map(GENE_MAP, UNGD)

# count grid
mapping_lib.read_countfiles(STATICDIR, COUNT_GLOB, UNGD)
mapping_lib.make_sample_tags(UNGD)

# locus_tag -> locus_len
mapping_lib.map_locus_tag_to_len(GENOME_FILE, UNGD)

# locus_tag -> bmk_ess
# locus_tag -> bmk_sick
mapping_lib.process_bmk_spreadsheet(JMPBMK_ANNOS, UNGD)

# variant -> is_oneoff
orig_map_frame = mapping_lib.get_mapping('variant', 'original', UNGD)
orig_map_frame.reset_index(inplace=True)
mapping_lib.map_variant_to_oneoff(orig_map_frame, UNGD)

# adapt OD data
mapping_lib.adapt_od_data(OD_DATA, UNGD)

# variant -> control
mapping_lib.make_variant_controltag_map(UNGD)
