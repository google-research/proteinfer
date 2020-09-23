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

"""Compute activations for trained model from input sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import functools
import gzip
import io
import itertools
import os
from typing import Dict, FrozenSet, Iterator, List, Text, Tuple

from absl import logging
import numpy as np
import pandas as pd
import utils
import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tqdm


def call_module(module, one_hots, row_lengths, signature):
  """Call a tf_hub.Module using the standard blundell signature.

  This expects that `module` has a signature named `signature` which conforms to
      ('sequence',
       'sequence_length') -> output

  To use an existing SavedModel
  file you may want to create a module_spec with
  `tensorflow_hub.saved_model_module.create_module_spec_from_saved_model`.

  Args:
    module: a tf_hub.Module to call.
    one_hots: a rank 3 tensor with one-hot encoded sequences of residues.
    row_lengths: a rank 1 tensor with sequence lengths.
    signature: the graph signature to validate and call.

  Returns:
    The output tensor of `module`.
  """
  if signature not in module.get_signature_names():
    raise ValueError('signature not in ' +
                     six.ensure_str(str(module.get_signature_names())) +
                     '. Was ' + six.ensure_str(signature) + '.')
  inputs = module.get_input_info_dict(signature=signature)
  expected_inputs = [
      'sequence',
      'sequence_length',
  ]
  if set(inputs.keys()) != set(expected_inputs):
    raise ValueError(
        'The signature_def does not have the expected inputs. Please '
        'reconfigure your saved model to only export signatures '
        'with sequence and length inputs. (Inputs were %s, expected %s)' %
        (str(inputs), str(expected_inputs)))

  outputs = module.get_output_info_dict(signature=signature)
  if len(outputs) > 1:
    raise ValueError('The signature_def given has more than one output. Please '
                     'reconfigure your saved model to only export signatures '
                     'with one output. (Outputs were %s)' % str(outputs))

  return list(
      module({
          'sequence': one_hots,
          'sequence_length': row_lengths,
      },
             signature=signature,
             as_dict=True).values())[0]


def in_graph_inferrer(sequences,
                      savedmodel_dir_path,
                      signature,
                      name_scope='inferrer'):
  """Add an in-graph inferrer to the active default graph.

  Additionally performs in-graph preprocessing, splitting strings, and encoding
  residues.

  Args:
    sequences: A tf.string Tensor representing a batch of sequences with shape
      [None].
    savedmodel_dir_path: Path to the directory with the SavedModel binary.
    signature: Name of the signature to use in `savedmodel_dir_path`. e.g.
      'pooled_representation'
    name_scope: Name scope to use for the loaded saved model.

  Returns:
    Output Tensor
  Raises:
    ValueError if signature does not conform to
      ('sequence',
       'sequence_length') -> output
    or if the specified signature is not present.
  """
  # Add variable to make it easier to refactor with multiple tags in future.
  tags = [tf.saved_model.tag_constants.SERVING]

  # Tokenization
  residues = tf.strings.unicode_split(sequences, 'UTF-8')
  # Convert to one-hots and pad.
  one_hots, row_lengths = utils.in_graph_residues_to_onehot(residues)
  module_spec = hub.saved_model_module.create_module_spec_from_saved_model(
      savedmodel_dir_path)
  module = hub.Module(module_spec, trainable=False, tags=tags, name=name_scope)
  return call_module(module, one_hots, row_lengths, signature)


@functools.lru_cache(maxsize=None)
def memoized_inferrer(
    savedmodel_dir_path,
    activation_type=tf.saved_model.signature_constants
    .DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    batch_size=64,
    use_tqdm=False,
    session_config=None,
    memoize_inference_results=False,
    use_latest_savedmodel=False,
):
  """Alternative constructor for Inferrer that is memoized."""
  return Inferrer(
      savedmodel_dir_path=savedmodel_dir_path,
      activation_type=activation_type,
      batch_size=batch_size,
      use_tqdm=use_tqdm,
      session_config=session_config,
      memoize_inference_results=memoize_inference_results,
      use_latest_savedmodel=use_latest_savedmodel,
  )


class Inferrer(object):
  """Uses a SavedModel to provide batched inference."""

  def __init__(
      self,
      savedmodel_dir_path,
      activation_type=tf.saved_model.signature_constants
      .DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      batch_size=64,
      use_tqdm=False,
      session_config=None,
      memoize_inference_results=False,
      use_latest_savedmodel=False,
  ):
    """Construct Inferrer.

    Args:
      savedmodel_dir_path: path to directory where a SavedModel pb or
        pbtxt is stored. The SavedModel must only have one input per signature
        and only one output per signature.
      activation_type: one of the keys in saved_model.signature_def.keys().
      batch_size: batch size to use for individual inference ops.
      use_tqdm: Whether to print progress using tqdm.
      session_config: tf.ConfigProto for tf.Session creation.
      memoize_inference_results: if True, calls to inference.get_activations
        will be memoized.
      use_latest_savedmodel: If True, the model will be loaded from
        latest_savedmodel_path_from_base_path(savedmodel_dir_path).

    Raises:
      ValueError: if activation_type is not the name of a signature_def in the
        SavedModel.
      ValueError: if SavedModel.signature_def[activation_type] has an input
        other than 'sequence'.
      ValueError: if SavedModel.signature_def[activation_type] has more than
        one output.
    """
    if use_latest_savedmodel:
      savedmodel_dir_path = latest_savedmodel_path_from_base_path(
          savedmodel_dir_path)
    self.batch_size = batch_size
    self._graph = tf.Graph()
    self._model_name_scope = 'inferrer'
    with self._graph.as_default():
      self._sequences = tf.placeholder(
          shape=[None], dtype=tf.string, name='sequences')
      self._fetch = in_graph_inferrer(
          self._sequences,
          savedmodel_dir_path,
          activation_type,
          name_scope=self._model_name_scope)
      self._sess = tf.Session(
          config=session_config if session_config else tf.ConfigProto())
      self._sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables(),
          tf.initializers.tables_initializer(),
      ])

    self._savedmodel_dir_path = savedmodel_dir_path
    self.activation_type = activation_type
    self._use_tqdm = use_tqdm
    if memoize_inference_results:
      self._get_activations_for_batch = self._get_activations_for_batch_memoized
    else:
      self._get_activations_for_batch = self._get_activations_for_batch_unmemoized

  def __repr__(self):
    return ('{} with feed tensors savedmodel_dir_path {} and '
            'activation_type {}').format(
                type(self).__name__, self._savedmodel_dir_path,
                self.activation_type)

  def _get_tensor_by_name(self, name):
    return self._graph.get_tensor_by_name('{}/{}'.format(
        self._model_name_scope, name))

  def _get_activations_for_batch_unmemoized(self,
                                            seqs,
                                            custom_tensor_to_retrieve=None):
    """Gets activations for each sequence in list_of_seqs.

      [
        [activation_1, activation_2, ...] # For list_of_seqs[0]
        [activation_1, activation_2, ...] # For list_of_seqs[1]
        ...
      ]

    In the case that the activations are the normalized probabilities that a
    sequence belongs to a class, entry `i, j` of
    `inferrer.get_activations(batch)` contains the probability that
    sequence `i` is in family `j`.

    Args:
      seqs: tuple of strings, with characters that are amino
        acids.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      np.array of floats containing the value from fetch_op.
    """
    if custom_tensor_to_retrieve:
      fetch = self._get_tensor_by_name(custom_tensor_to_retrieve)
    else:
      fetch = self._fetch
    with self._graph.as_default():
      return self._sess.run(fetch, {self._sequences: seqs})

  @functools.lru_cache(maxsize=None)
  def _get_activations_for_batch_memoized(self,
                                          seqs,
                                          custom_tensor_to_retrieve=None):
    return self._get_activations_for_batch_unmemoized(
        seqs, custom_tensor_to_retrieve)

  def get_activations(self, list_of_seqs, custom_tensor_to_retrieve=None):
    """Gets activations where batching may be needed to avoid OOM.

    Inputs are strings of amino acids, outputs are activations from the network.

    Args:
      list_of_seqs: iterable of strings as input for inference.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      concatenated numpy array of activations with shape [num_of_seqs, ...]
    """
    np_seqs = np.array(list_of_seqs, dtype=np.str_)
    if np_seqs.size == 0:
      return np.array([], dtype=float)

    if len(np_seqs.shape) != 1:
      raise ValueError('`list_of_seqs` should be convertible to a numpy vector '
                       'of strings. Got {}'.format(np_seqs))

    logging.debug('Predicting for %d sequences', len(list_of_seqs))

    lengths = np.array([len(seq) for seq in np_seqs])
    # Sort by reverse length, so that the longest element is first.
    # This is because the longest element can cause memory issues, and we'd like
    # to fail-fast in this case.
    sorter = np.argsort(lengths)[::-1]
    # The inverse of a permutation A is the permutation B such that B(A) is the
    # the identity permutation (a sorted list).
    reverser = np.argsort(sorter)

    activation_list = []
    batches = np.array_split(np_seqs[sorter],
                             np.ceil(len(np_seqs) / self.batch_size))
    if self._use_tqdm:
      batches = tqdm.tqdm(
          batches,
          position=0,
          desc='Annotating batches of sequences',
          leave=True,
          dynamic_ncols=True)
    for batch in batches:
      batch_activations = self._get_activations_for_batch(
          tuple(batch), custom_tensor_to_retrieve=custom_tensor_to_retrieve)

      activation_list.append(batch_activations)

    activations = np.concatenate(activation_list, axis=0)[reverser]

    return activations

  def get_variable(self, variable_name):
    """Gets the value of a variable from the graph.

    Args:
      variable_name: string name for retrieval. E.g. "vocab_name:0"

    Returns:
      output from TensorFlow from attempt to retrieve this value.
    """
    with self._graph.as_default():
      return self._sess.run(self._get_tensor_by_name(variable_name))


def latest_savedmodel_path_from_base_path(base_path):
  """Get the most recent savedmodel from a base directory path."""

  protein_export_base_path = os.path.join(base_path, 'export/protein_exporter')

  suffixes = [
      x for x in tf.io.gfile.listdir(protein_export_base_path)
      if 'temp-' not in x
  ]

  if not suffixes:
    raise ValueError('No SavedModels found in %s' % protein_export_base_path)

  # Sort by suffix to take the model corresponding the most
  # recent training step.
  return os.path.join(protein_export_base_path, sorted(suffixes)[-1])


def predictions_for_df(df, inferrer):
  """Returns df with column that's the activations for each sequence.

  Args:
    df: DataFrame with columns 'sequence' and 'sequence_name'.
    inferrer: inferrer.

  Returns:
    pd.DataFrame with columns 'sequence_name', 'predicted_label', and
    'predictions'. 'predictions' has type np.ndarray, whose shape depends on
    inferrer.activation_type.
  """
  working_df = df.copy()
  working_df['predictions'] = inferrer.get_activations(
      working_df.sequence.values).tolist()
  return working_df


def serialize_inference_result(sequence_name,
                               activations):
  """Serializes an inference result.

  This function is the opposite of deserialize_inference_result.

  The full format returned is a
  base-64 encoded ( np compressed array of ( dict of (seq_name: activations))))

  Benefits of this setup:
  - Takes advantage of np compression.
  - Avoids explicit use of pickle (e.g. np.save(allow_pickle)).
  - Is somewhat agnostic to the dictionary contents
    (i.e. you can put whatever you want in the dictionary if we wanted to reuse
    this serialization format)
  - No protos, so no build dependencies for colab.
  - Entries are serialized row-wise, so they're easy to iterate through, and
    it's possible to decode them on the fly.

  Args:
    sequence_name: sequence name.
    activations: np.ndarray.

  Returns:
    encoded/serialized version of sequence_name and activations.
  """
  with io.BytesIO() as bytes_io:
    np.savez_compressed(bytes_io, **{sequence_name: activations})
    return base64.b64encode(bytes_io.getvalue())


def deserialize_inference_result(results_b64):
  """Deserializes an inference result.

  This function is the opposite of serialize_inference_result.

  The full format expected is a
  base-64 encoded ( np compressed array of ( dict of (seq_name: activations))))

  Benefits of this setup:
  - Takes advantage of np compression.
  - Avoids explicit use of pickle (e.g. np.save(allow_pickle)).
  - Is somewhat agnostic to the dictionary contents
    (i.e. you can put whatever you want in the dictionary if we wanted to reuse
    this serialization format)
  - No protos, so no build dependencies for colab.
  - Entries are serialized row-wise, so they're easy to iterate through, and
    it's possible to decode them on the fly.

  Args:
    results_b64: bytes with the above contents.

  Returns:
    tuple of sequence_name, np.ndarray (the activations).

  Raises:
    ValueError if the structured np.array containing the activations doesn't
    have exactly 1 element.
  """
  bytes_io = io.BytesIO(base64.b64decode(results_b64))
  single_pred_dict = dict(np.load(bytes_io))
  if len(single_pred_dict) != 1:
    raise ValueError('Expected exactly one object in the structured np array. '
                     f'Saw {len(single_pred_dict)}')
  sequence_name = list(single_pred_dict.keys())[0]
  activations = list(single_pred_dict.values())[0]
  return sequence_name, activations


def parse_shard(shard_path):
  """Parses file of gzipped, newline-separated inference results.

  The contents of each line are expected to be serialized as in
  `serialize_inference_result` above.

  Args:
    shard_path: file path.

  Yields:
    Tuple of (accession, activation).
  """
  with tf.io.gfile.GFile(shard_path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as f_gz:
      for line in f_gz:  # Line-by-line.
        yield deserialize_inference_result(line)


def parse_all_shards(shard_dir_path):
  """Parses directory of files of gzipped, newline-separated inference results.

  The contents of each line are expected to be serialized as in
  `serialize_inference_result` above.

  Args:
    shard_dir_path: path to directory containing shards.

  Returns:
    DataFrame with columns sequence_name (str); predictions (rank 1 np.ndarray
    of activations).
  """
  files_to_process = utils.absolute_paths_of_files_in_dir(shard_dir_path)
  list_of_shard_results = [parse_shard(f) for f in files_to_process]
  return pd.DataFrame(
      list(itertools.chain(*list_of_shard_results)),
      columns=['sequence_name', 'predictions'])


def get_preds_at_or_above_threshold(input_df,
                                    inferrer_list,
                                    threshold):
  """Runs ensembled inference; returns dataframe of filtered inference results.

  Includes predictions >= threshold.

  Because more than one label can be predicted for a sequence, the same
  sequence_name may appear in multiple output rows

  Args:
    input_df: pd.DataFrame with columns sequence (str), sequence_name (str).
    inferrer_list: list of ensemble elements.
    threshold: float. Keep inference results above this threshold.

  Returns:
    pd.DataFrame with columns sequence_name (str), predicted_label (str), and
    confidence (float). `sequence_name`s are sorted in the original order they
    came in.
  """
  if threshold == 0.:
    raise ValueError('The given threshold was 0. Please supply a '
                     'value between 0 (exclusive) and 1 (inclusive). A value '
                     'of zero will report every label for every protein.')
  predictions = np.mean([
      inferrer.get_activations(input_df.sequence.values.tolist())
      for inferrer in inferrer_list
  ],
                        axis=0)
  cnn_label_vocab = inferrer_list[0].get_variable('label_vocab:0').astype(str)

  output_dict = {'sequence_name': [], 'predicted_label': [], 'confidence': []}

  for idx, protein in enumerate(predictions):
    proteins_above_threshold = protein >= threshold
    labels_predicted = cnn_label_vocab[proteins_above_threshold]
    for label, confidence in zip(labels_predicted,
                                 protein[proteins_above_threshold]):
      output_dict['sequence_name'].append(input_df.sequence_name.values[idx])
      output_dict['predicted_label'].append(label)
      output_dict['confidence'].append(confidence)

  return pd.DataFrame(output_dict)
