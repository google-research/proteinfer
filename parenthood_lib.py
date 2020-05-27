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

# Lint as: python3
"""Utilities for parsing EC and GO labels, and finding their parents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
from typing import Dict, FrozenSet, List, Optional, Set, Text, Collection
import pandas as pd
import tensorflow.compat.v1 as tf

APPLICABLE_LABEL_JSON_PATH = 'parenthood.json.gz'


def get_applicable_label_dict(
    path = APPLICABLE_LABEL_JSON_PATH):
  with tf.io.gfile.GFile(path, 'r') as f:
    with gzip.GzipFile(fileobj=f, mode='r') as gzip_file:
      return json.load(gzip_file)


def reverse_map(
    applicable_label_dict,
    label_vocab = None):
  """Flip parenthood dict to map parents to children."""
  # This is technically the entire transitive closure, so it is safe for DAGs
  # (e.g. GO labels).

  children = collections.defaultdict(set)
  for child, parents in applicable_label_dict.items():
    # Avoid adding children which don't appear in the vocab.
    if label_vocab is None or child in label_vocab:
      for parent in parents:
        children[parent].add(child)
  return {k: frozenset(v) for k, v in children.items()}


def is_implied_by_something_else(
    current_label,
    reversed_normalizer,
    all_labels_for_protein,
):
  """Returns whether the current label is implied by other labels for protein.

  Args:
    current_label: label about which we're asking "is this implied by some other
      label for this protein?"
    reversed_normalizer: output of reverse_map(label_normalizer). Helps this
      function run fast.
    all_labels_for_protein: set of all labels given to protein.

  Returns:
    bool
  """
  all_labels_for_protein_without_current = all_labels_for_protein - frozenset(
      [current_label])

  children_of_current_label = reversed_normalizer[current_label]

  # Most labels imply themselves; remove.
  children_of_current_label = children_of_current_label - frozenset(
      [current_label])

  return len(  # pylint: disable=g-explicit-length-test
      children_of_current_label.intersection(
          all_labels_for_protein_without_current)) > 0


def _filter_label_set_to_most_specific(
    label_set,
    reversed_normalizer):
  """Filters label set to most specific.

  Args:
    label_set: set of all labels given to protein.
    reversed_normalizer: output of reverse_map(label_normalizer). Helps this
      function run fast.

  Returns:
    Filtered set of labels.
  """
  return frozenset([
      l for l in label_set
      if not is_implied_by_something_else(l, reversed_normalizer, label_set)
  ])


def filter_labels_to_most_specific(
    df,
    normalizer,
    column_to_filter = 'predicted_label',
):
  """Filter labels given to each protein to the most specific label.

  Useful for labels like GO, where we predict a ton of labels, and we only
  want to look at the most informative labels.

  Args:
    df: pd.DataFrame with column `column_to_filter`.
    normalizer: label normalizer.
    column_to_filter: name of column in df.

  Returns:
    pd.DataFrame with column `column_to_filter`.
  """
  reversed_normalizer = reverse_map(normalizer)

  working_df = df.copy()
  working_df[column_to_filter] = working_df[column_to_filter].apply(
      lambda label_set: _filter_label_set_to_most_specific(  # pylint: disable=g-long-lambda
          label_set, reversed_normalizer))
  return working_df
