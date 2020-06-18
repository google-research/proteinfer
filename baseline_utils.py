# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Utilities for evaluating baseline implementations of function annotation."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import re
import typing
from typing import Any, Dict, FrozenSet, List, Set, Text

import numpy as np
import pandas as pd
import six
from six.moves import zip
import tensorflow.compat.v1 as tf
import tqdm

_BLAST_OUTPUT_COLUMNS = [
    'query_id',
    'subject_id',
    'percent_seq_identity',
    'alignment_length',
    'n_mismatches',
    'n_gap_opens',
    'query_alignment_start',
    'query_alignment_end',
    'subject_alignment_start',
    'subject_alignment_end',
    'expected_value',
    'bit_score',
]

_MERGE_COL_NAME = 'merge_col'

# e.g. >accession="A0A0PX123"	labels="PF00001,PF00002"
# Notice optional fasta header character '>' at the beginning.
_SEQUENCE_HEADER_RE = re.compile(r'^>?accession="(\w+)"\tlabels="(.*)"')

# Format of query sequence field for interproscan.
_INTERPROSCAN_ACCESSION_REGEX = re.compile(r'accession="(\w+)"')


def _get_sequence_name_from_sequence_header(s):
  """Returns sequence name from fasta header.

  Args:
    s: Fasta header input string. Format of sequence headers is
      >accession_"label1,label2". The initial '>' character is optional.
  """
  return _SEQUENCE_HEADER_RE.findall(s)[0][0]  # 0th match, 0th capture group.


def _get_labels_from_sequence_header(s):
  r"""Returns labels from fasta header.

  Args:
    s: Fasta header input string. Format of sequence headers is
      >accession_"label1,label2". The initial '>' character is optional.
  """
  # 0th match, 1th capture group.
  regex_match = _SEQUENCE_HEADER_RE.findall(s)[0][1]
  if regex_match == '':  # pylint: disable=g-explicit-bool-comparison
    # If the regex match is the empty string, we just want an empty set, as
    # there are no labels.
    return set()
  return set(six.ensure_str(regex_match).split(','))


class BlastResult(
    typing.NamedTuple('BlastResult', (
        ('sequence_name', Text),
        ('closest_sequence', Text),
        ('predicted_label', Set[Text]),
        ('percent_seq_identity', float),
        ('e_value', float),
        ('bit_score', float),
    ))):
  """Result of parsing BLAST output."""

  @staticmethod
  def from_string(s, ground_truth_lookup):
    """Constructs a BlastResult from a line in blast tsv output."""
    tab_split = six.ensure_str(s).split('\t')
    query_sequence = six.ensure_str(tab_split[0]).split('"')[1]
    closest_sequence = six.ensure_str(tab_split[1]).split('"')[1]

    percent_seq_identity = float(tab_split[2])
    e_value = float(tab_split[10])
    bit_score = float(tab_split[11])

    predicted_label = ground_truth_lookup[closest_sequence]

    return BlastResult(
        sequence_name=query_sequence,
        closest_sequence=closest_sequence,
        predicted_label=predicted_label,
        percent_seq_identity=percent_seq_identity,
        e_value=e_value,
        bit_score=bit_score)


class InterproScanResult(
    typing.NamedTuple('InterproScanResult', (('sequence_name', Text),
                                             ('predicted_label', Set[Text])))):
  """Result of parsing InterproScan output."""

  @staticmethod
  def from_string(s):
    tab_split = six.ensure_str(s).rstrip('\n').split('\t')
    if len(tab_split) < 14 or tab_split[13] == '':  # pylint: disable=g-explicit-bool-comparison
      go_labels = set()
    else:
      go_labels = set(six.ensure_str(tab_split[13]).split('|'))

    return InterproScanResult(
        sequence_name=_INTERPROSCAN_ACCESSION_REGEX.findall(tab_split[0])[0],
        predicted_label=go_labels)


def load_ground_truth(filename):
  """Returns dataframe of ground truth from FASTA file.

  Documentation on how to produce this FASTA file is available at
  https://docs.google.com/document/d/1uB8J_a1cURIeVNqa5SOJRBBtimI1nOHbU1NbN43Xr0s

  Args:
    filename: str. Path to FASTA file. Fasta header input string. Format of
      sequence headers is like >accession="A0A0PX123"   labels="PF00001,PF00002"

  Returns:
    pd.DataFrame with columns sequence_name (str), sequence (str),
    true_label(Set[str]).
  """
  dict_for_making_df = {'sequence_name': [], 'true_label': [], 'sequence': []}

  with tf.io.gfile.GFile(filename) as f:
    for line in tqdm.tqdm(f, position=0):
      line = six.ensure_str(line)
      if line.startswith('>'):
        dict_for_making_df['sequence_name'].append(
            _get_sequence_name_from_sequence_header(line))
        dict_for_making_df['true_label'].append(
            _get_labels_from_sequence_header(line))
      else:
        dict_for_making_df['sequence'].append(line.rstrip())

  all_test_seqs = pd.DataFrame(dict_for_making_df)

  return all_test_seqs


def _set_predicted_labels_missing(row):
  """Adds empty set as predicted labels for a DataFrame row."""
  if row[_MERGE_COL_NAME] == 'right_only':
    return set()
  else:
    return row['predicted_labels']


def _fillna(df, column_name, value):
  """Replacement for df.fillna that works with non-indexable values.

  https://github.com/pandas-dev/pandas/issues/21329

  Args:
    df: pd.DataFrame with column `column_name`.
    column_name: str.
    value: value to set when column's value is NaN.
  """
  df[column_name] = [
      (x if isinstance(x, (set, frozenset)) else value) for x in df[column_name]
  ]


def _blast_row_to_confidence_array(
    row,
    label_vocab,
    label_to_vocab_index,
):
  """Returns a confidence array of preds (predicted labels get bit_score)."""
  arr = np.zeros_like(label_vocab, np.float32)
  indexes_to_update = np.array(
      [label_to_vocab_index[l] for l in row.predicted_label], dtype=np.int32)
  arr[indexes_to_update] = row.bit_score
  return arr


def load_blast_output(filename, label_vocab,
                      training_data_ground_truth,
                      test_data_ground_truth):
  """Load a file containing tsv-separated blast output.

  Args:
    filename: str. Path to output of blast.
    label_vocab: np.array of string (labels). Used to compute the output column
      `predictions`.
    training_data_ground_truth: pd.DataFrame. The output of `load_ground_truth`.
      Used to compute predicted labels, by using the labels on the training data
      for the best match, given by blast.
    test_data_ground_truth: pd.DataFrame. The output of `load_ground_truth`.
      Used to compute the true labels.

  Returns:
    pd.DataFrame with columns
    sequence_name (str);
    true_label (Set[str]);
    predicted_label (Set[str]);
    predictions (np.array of length len(label_vocab), filled with the bit score
      of the closest match). A value of 0 is used for sequences with no blast
      calls;
    percent_seq_identity (float, between 0 and 100);
    e_value (float, a measure of confidence. Lower is more confident.).
      A value of NaN is used for missing blast calls;
    bit_score (float, a measure of confidence. Higher is more confident.).
      A value of 0 is used for sequences with no blast calls.

    For more information on e-value and bit score, see
    http://www.metagenomics.wiki/tools/blast/evalue
  """
  training_data_ground_truth_lookup = dict(
      list(
          zip(training_data_ground_truth.sequence_name,
              training_data_ground_truth.true_label)))

  blast_output_namedtuples = []
  with tf.io.gfile.GFile(filename) as f:
    for line in tqdm.tqdm(f, position=0):
      blast_output_namedtuples.append(
          BlastResult.from_string(
              line, training_data_ground_truth_lookup))

  blast_output = pd.DataFrame.from_records(
      blast_output_namedtuples, columns=BlastResult._fields)

  # Add in ground truth to get all sequences (including those missed by BLAST).
  blast_output = blast_output.merge(
      test_data_ground_truth,
      on='sequence_name',
      how='right',
      indicator=_MERGE_COL_NAME)

  label_to_vocab_index = {l: i for i, l in enumerate(label_vocab)}
  # For all sequences for which we don't have predictions, set the predictions
  # to the empty set.
  _fillna(blast_output, 'predicted_label', frozenset())

  # A bit score of 0 is a "zero confidence" score.
  blast_output['bit_score'].fillna(0., inplace=True)

  blast_output['predictions'] = blast_output.apply(
      axis='columns',
      func=lambda row: _blast_row_to_confidence_array(  # pylint: disable=g-long-lambda
          row, label_vocab, label_to_vocab_index))

  # Clean up output columns.
  blast_output.drop(inplace=True, columns=[_MERGE_COL_NAME, 'sequence'])

  return blast_output


def _pad_ec_label_with_hyphens(label):
  """E.g. Given 'EC:1', returns 'EC:1.-.-.-'."""
  to_return = label
  while to_return.count('.') != 3:
    to_return += '.-'
  return to_return


def limit_set_of_labels(df, acceptable_labels,
                        column_to_limit):
  """Limits the set of things in df[column_to_limit to acceptable_labels."""
  working_df = df.copy()

  working_df[column_to_limit] = working_df[column_to_limit].apply(
      acceptable_labels.intersection)
  return working_df


def limit_labels_for_label_normalizer(
    label_normalizer,
    acceptable_labels):
  """Limits keys and values in label_normalizer to acceptable labels."""
  limited_label_normalizer = {}
  for k, v in label_normalizer.items():
    if k in acceptable_labels:
      limited_label_normalizer[k] = sorted(
          acceptable_labels.intersection(set(v)))

  return limited_label_normalizer


def load_interproscan_output(
    interproscan_output_filename,
    test_data_ground_truth,
):
  """Loads tab-separated output from interproscan.

  Args:
    interproscan_output_filename: file path to tsv interproscan output.
    test_data_ground_truth: pd.DataFrame. The output of `load_ground_truth`.
      Used to compute the true labels.

  Returns:
    pd.DataFrame with output columns 'sequence_name' (Text), predicted_label
    (frozenset).
  """
  interproscan_output_namedtuples = []
  with tf.io.gfile.GFile(interproscan_output_filename) as f:
    for line in tqdm.tqdm(f, position=0):
      interproscan_output_namedtuples.append(
          InterproScanResult.from_string(line))

  interproscan_output = pd.DataFrame.from_records(
      interproscan_output_namedtuples, columns=InterproScanResult._fields)

  collapsed_interproscan_output_dict = {
      'sequence_name': [],
      'predicted_label': []
  }

  # There is more than one output per accession because interproscan supports
  # many analyses of each sequence, and so all the go labels need to be
  # collapsed.
  for grouping_key, group in tqdm.tqdm(
      interproscan_output.groupby('sequence_name'), position=0):
    collapsed_interproscan_output_dict['sequence_name'].append(grouping_key)

    collapsed_interproscan_output_dict['predicted_label'].append(
        frozenset().union(*group.predicted_label.values.tolist()))

  collapsed_interproscan_output = pd.DataFrame(
      collapsed_interproscan_output_dict)

  # Add in ground truth to get all sequences (including those missed by BLAST).
  interproscan_output = collapsed_interproscan_output.merge(
      test_data_ground_truth,
      on='sequence_name',
      how='right',
      indicator=_MERGE_COL_NAME)

  # For all sequences for which we don't have predictions, set the predictions
  # to the empty set.
  _fillna(interproscan_output, 'predicted_label', frozenset())

  # Clean up output columns.
  interproscan_output.drop(inplace=True, columns=[_MERGE_COL_NAME, 'sequence'])

  return interproscan_output
