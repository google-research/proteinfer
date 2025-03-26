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

import functools
import gzip
import json
import os
import tarfile
from typing import (Callable, List, Optional, Text)
import urllib

import numpy as np
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import lookup as contrib_lookup
import tqdm


AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

_PFAM_GAP_CHARACTER = '.'

# Other characters representing amino-acids not in AMINO_ACID_VOCABULARY.
_ADDITIONAL_AA_VOCABULARY = [
    # Substitutions
    'U',
    'O',
    # Ambiguous Characters
    'B',
    'Z',
    'X',
    # Gap Character
    _PFAM_GAP_CHARACTER
]

# Vocab of all possible tokens in a valid input sequence
FULL_RESIDUE_VOCAB = AMINO_ACID_VOCABULARY + _ADDITIONAL_AA_VOCABULARY

# Map AA characters to their index in FULL_RESIDUE_VOCAB.
_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(FULL_RESIDUE_VOCAB)}

OSS_ZIPPED_MODELS_ROOT_URL = 'https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/'
_OSS_PFAM_ZIPPED_MODELS_URL_BASE = OSS_ZIPPED_MODELS_ROOT_URL + 'noxpd2_cnn_swissprot_pfam_random_swiss-cnn_for_swissprot_pfam_random-'
_OSS_EC_ZIPPED_MODELS_URL_BASE = OSS_ZIPPED_MODELS_ROOT_URL + 'noxpd2_cnn_swissprot_ec_random_swiss-cnn_for_swissprot_ec_random-'
_OSS_GO_ZIPPED_MODELS_URL_BASE = OSS_ZIPPED_MODELS_ROOT_URL + 'noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-'

MAX_NUM_ENSEMBLE_ELS_FOR_INFERENCE = 5

PARENTHOOD_FILE_URL = 'https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/colab_support/parenthood.json.gz'
LABEL_DESCRIPTION_URL = 'https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/colab_support/label_descriptions.json.gz'
INSTALLED_PARENTHOOD_FILE_NAME = 'parenthood.json.gz'
INSTALLED_LABEL_DESCRIPTION_FILE_NAME = 'label_descriptions.json.gz'

# pyformat: disable
PFAM_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    '13703743', '13703976', '13704038', '13704097', '13704156', '13705318',
    '13705635', '13705680', '13705733', '13705759', '13705805', '13706336',
    '13707555', '13707708', '13707739', '13707862', '13708715', '13708866',
    '13709033', '13709258', '13709363', '13709600', '13709998', '13710430',
    '13711765', '13729975', '13730021', '13730128', '13730776', '13730885',
    '13731191', '13731551', '13731565', '13731695', '13732031',
]

EC_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    '13703966', '13704083', '13704104', '13704130', '13705280', '13705675',
    '13705786', '13705802', '13705819', '13705839', '13706239', '13706986',
    '13707020', '13707589', '13707925', '13708369', '13708672', '13708706',
    '13708740', '13708951', '13709242', '13709584', '13709983', '13710037',
    '13711670', '13729344', '13730041', '13730097', '13730679', '13730876',
    '13730909', '13731218', '13731588', '13731728', '13731976',
]

GO_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    '13703706', '13703742', '13703997', '13704131', '13705631', '13705668',
    '13705677', '13705689', '13705708', '13705728', '13706170', '13706215',
    '13707414', '13707438', '13707732', '13708169', '13708676', '13708925',
    '13708995', '13709052', '13709428', '13709589', '13710370', '13710418',
    '13711677', '13729352', '13730011', '13730387', '13730746', '13730766',
    '13730958', '13731179', '13731598', '13731645', '13732022',
]
# pyformat: enable

OSS_PFAM_ZIPPED_MODELS_URLS = [
    '{}{}.tar.gz'.format(_OSS_PFAM_ZIPPED_MODELS_URL_BASE, p)
    for p in PFAM_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS
]
OSS_EC_ZIPPED_MODELS_URLS = [
    '{}{}.tar.gz'.format(_OSS_EC_ZIPPED_MODELS_URL_BASE, p)
    for p in EC_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS
]
OSS_GO_ZIPPED_MODELS_URLS = [
    '{}{}.tar.gz'.format(_OSS_GO_ZIPPED_MODELS_URL_BASE, p)
    for p in GO_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS
]


def residues_to_indices(amino_acid_residues):
  return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]


def normalize_sequence_to_blosum_characters(seq):
  """Make substitutions, since blosum62 doesn't include amino acids U and O.

  We take the advice from here for the appropriate substitutions:
  https://www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/multalignviewer/multalignviewer.html

  Args:
    seq: amino acid sequence. A string.

  Returns:
    An amino acid sequence string that's compatible with the blosum substitution
    matrix.
  """
  return seq.replace('U', 'C').replace('O', 'X')


@functools.lru_cache(maxsize=1)
def _build_one_hot_encodings():
  """Create array of one-hot embeddings.

  Row `i` of the returned array corresponds to the one-hot embedding of amino
    acid FULL_RESIDUE_VOCAB[i].

  Returns:
    np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
  """
  base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
  to_aa_index = AMINO_ACID_VOCABULARY.index

  special_mappings = {
      'B':
          .5 *
          (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
      'Z':
          .5 *
          (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
      'X':
          np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
      _PFAM_GAP_CHARACTER:
          np.zeros(len(AMINO_ACID_VOCABULARY)),
  }
  special_mappings['U'] = base_encodings[to_aa_index('C')]
  special_mappings['O'] = special_mappings['X']
  special_encodings = np.array(
      [special_mappings[c] for c in _ADDITIONAL_AA_VOCABULARY])
  return np.concatenate((base_encodings, special_encodings), axis=0)


def residues_to_one_hot(amino_acid_residues):
  """Given a sequence of amino acids, return one hot array.

  Supports ambiguous amino acid characters B, Z, and X by distributing evenly
  over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].

  Supports rare amino acids by appropriately substituting. See
  normalize_sequence_to_blosum_characters for more information.

  Supports gaps and pads with the '.' and '-' characters; which are mapped to
  the zero vector.

  Args:
    amino_acid_residues: string. consisting of characters from
      AMINO_ACID_VOCABULARY

  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).

  Raises:
    KeyError: if amino_acid_residues has a character not in FULL_RESIDUE_VOCAB.
  """
  residue_encodings = _build_one_hot_encodings()
  int_sequence = residues_to_indices(amino_acid_residues)
  return residue_encodings[int_sequence]


def fasta_indexer():
  """Get a function for converting tokenized protein strings to indices."""
  mapping = tf.constant(FULL_RESIDUE_VOCAB)
  table = contrib_lookup.index_table_from_tensor(mapping)

  def mapper(residues):
    return tf.ragged.map_flat_values(table.lookup, residues)

  return mapper


def fasta_encoder():
  """Get a function for converting indexed amino acids to one-hot encodings."""
  encoded = residues_to_one_hot(''.join(FULL_RESIDUE_VOCAB))
  one_hot_embeddings = tf.constant(encoded, dtype=tf.float32)

  def mapper(residues):
    return tf.ragged.map_flat_values(
        tf.gather, indices=residues, params=one_hot_embeddings)

  return mapper


def in_graph_residues_to_onehot(residues):
  """Performs mapping in `residues_to_one_hot` in-graph.

  Args:
    residues: A tf.RaggedTensor with tokenized residues.

  Returns:
    A tuple of tensors (one_hots, row_lengths):
      `one_hots` is a Tensor<shape=[None, None, len(AMINO_ACID_VOCABULARY)],
                             dtype=tf.float32>
       that contains a one_hot encoding of the residues and pads out all the
       residues to the max sequence length in the batch by 0s.
       `row_lengths` is a Tensor<shape=[None], dtype=tf.int32> with the length
       of the unpadded sequences from residues.

  Raises:
    tf.errors.InvalidArgumentError: if `residues` contains a token not in
    `FULL_RESIDUE_VOCAB`.
  """
  ragged_one_hots = fasta_encoder()(fasta_indexer()(residues))
  return (ragged_one_hots.to_tensor(default_value=0),
          tf.cast(ragged_one_hots.row_lengths(), dtype=tf.int32))


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


def absolute_paths_of_files_in_dir(dir_path):
  files = os.listdir(dir_path)
  return sorted([os.path.join(dir_path, f) for f in files])


def load_gz_json(path):
  with open(path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as gzip_file:
      return json.load(gzip_file)


def fetch_oss_pretrained_models(
    model_type,
    output_dir_path,
    num_ensemble_elements = None):
  """Fetch, unzip, and untar a number of models to output_dir_path.

  Does not store the tar.gz versions, just the unzipped ones.

  Args:
    model_type: one of Pfam, EC, or GO.
    output_dir_path: output directory to which ensemble elements should be
      written.
    num_ensemble_elements: number of elements to fetch. If None, fetch all
      available.

  Raises:
    ValueError if model_type is invalid, or num_ensemble_elements is too large.
  """
  if model_type.lower() == 'pfam':
    absolute_model_urls = OSS_PFAM_ZIPPED_MODELS_URLS
  elif model_type.lower() == 'ec':
    absolute_model_urls = OSS_EC_ZIPPED_MODELS_URLS
  elif model_type.lower() == 'go':
    absolute_model_urls = OSS_GO_ZIPPED_MODELS_URLS
  else:
    raise ValueError(
        'Given model type {} was not valid. Valid model types are {}'.format(
            model_type, ['Pfam', 'EC', 'GO']))

  num_ensemble_elements = num_ensemble_elements if num_ensemble_elements is not None else len(
      absolute_model_urls)

  if num_ensemble_elements > len(absolute_model_urls):
    raise ValueError(
        'Requested {} ensemble elements, but only {} were available.'.format(
            num_ensemble_elements, len(absolute_model_urls)))

  absolute_model_urls = absolute_model_urls[:num_ensemble_elements]
  for absolute_url in tqdm.tqdm(
      absolute_model_urls,
      desc='Downloading and unzipping {} models to {}'.format(
          model_type, output_dir_path),
      position=0,
      leave=True):
    # TODO(mlbileschi): consider parallelizing to make faster.

    relative_file_name = os.path.basename(os.path.normpath(absolute_url))
    output_path = os.path.join(output_dir_path, relative_file_name)

    with urllib.request.urlopen(absolute_url) as url_contents:
      with tarfile.open(fileobj=url_contents, mode='r|gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, output_dir_path)
