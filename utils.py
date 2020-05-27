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

"""Utility functions for protein models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]


def calculate_bucket_batch_sizes(bucket_boundaries, max_expected_sequence_size,
                                 largest_batch_size):
  """Calculated batch sizes for each bucket given a set of boundaries.

  Sequences in the smallest sized bucket will get a batch_size of
  largest_batch_size and larger buckets will have smaller batch sizes  in
  proportion to their maximum sequence length to ensure that they do not use too
  much memory.

  E.g. for bucket_boundaries of [5, 10, 20, 40], max_expected_size of 100
  and largest_batch_size of 50, expected_bucket_sizes are [50, 25, 12, 6, 2].

  Args:
    bucket_boundaries: list of positions of bucket boundaries
    max_expected_sequence_size: largest expected sequence, used to calculate
      sizes
    largest_batch_size: batch_size for largest batches.

  Returns:
    batch_sizes as list
  """
  first_max_size = bucket_boundaries[0]
  bucket_relative_batch_sizes = [
      (first_max_size / x)
      for x in bucket_boundaries + [max_expected_sequence_size]
  ]
  bucket_absolute_batch_sizes = [
      int(x * largest_batch_size) for x in bucket_relative_batch_sizes
  ]
  if min(bucket_absolute_batch_sizes) == 0:
    raise ValueError(
        'There would be a batch size of 0 during bucketing, which is not '
        'allowed. Bucket boundaries passed in were: %s, leading to batch sizes of: %s'
        % (bucket_boundaries, bucket_absolute_batch_sizes))
  return bucket_absolute_batch_sizes


def residues_to_one_hot(residue_sequence):
  """Given a sequence of amino acids, return one hot array.

  Args:
    residue_sequence: string. consisting of characters from
      AMINO_ACID_VOCABULARY

  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).

  Raises:
    ValueError: if sparse_amino_acid has a character not in the vocabulary.
  """
  onehots = np.zeros((len(residue_sequence), len(AMINO_ACID_VOCABULARY)))

  for i, char in enumerate(residue_sequence):
    if char in AMINO_ACID_VOCABULARY:
      onehots[i, AMINO_ACID_VOCABULARY.index(char)] = 1
    else:
      raise ValueError('Could not one-hot code character {}'.format(char))
  return onehots


def batch_iterable(iterable, batch_size):
  """Yields batches from an iterable.

  If the number of elements in the iterator is not a multiple of batch size,
  the last batch will have fewer elements.

  Args:
    iterable: a potentially infinite iterable.
    batch_size: the size of batches to return.

  Yields:
    array of length batch_size, containing elements, in order, from iterable.

  Raises:
    ValueError: if batch_size < 1.
  """
  if batch_size < 1:
    raise ValueError(
        'Cannot have a batch size of less than 1. Received: {}'.format(
            batch_size))

  current = []
  for item in iterable:
    if len(current) == batch_size:
      yield current
      current = []
    current.append(item)

  # Prevent yielding an empty batch. Instead, prefer to end the generation.
  if current:
    yield current


def pad_one_hot(one_hot, length):
  if length < one_hot.shape[0]:
    raise ValueError("The padding value must be longer than the one-hot's 0th "
                     'dimension. Padding value is ' + str(length) + ' '
                     'and one-hot shape is ' + str(one_hot.shape))
  padding = np.zeros((length - one_hot.shape[0], len(AMINO_ACID_VOCABULARY)))
  return np.append(one_hot, padding, axis=0)


def make_padded_np_array(ragged_arrays):
  """Converts ragged array of one-hot amino acids to constant-length np.array.

  Args:
    ragged_arrays: list of list of int. Each entry in the list is a one-hot
      encoded protein, where each entry corresponds to an amino acid.

  Returns:
    np.array of int, shape (len(ragged_arrays),
      len(longest_array_in_ragged_arrays), len(AMINO_ACID_VOCABULARY)).
  """
  max_array_length = max(len(a) for a in ragged_arrays)
  return np.array([
      pad_one_hot(ragged_array, max_array_length)
      for ragged_array in ragged_arrays
  ])
