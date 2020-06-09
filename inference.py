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
import gzip
import io
import itertools
from typing import Iterator, Text, Tuple

from absl import logging
import numpy as np
import pandas as pd
import protein_dataset
import utils
import tensorflow.compat.v1 as tf
import tqdm


class Inferrer(object):
  """Fetches an op's value given features."""

  def __init__(self,
               savedmodel_dir_path,
               activation_type=tf.saved_model.signature_constants
               .DEFAULT_SERVING_SIGNATURE_DEF_KEY,
               batch_size=64,
               use_tqdm=False):
    """Construct Inferrer.

    Args:
      savedmodel_dir_path: path to directory where a SavedModel pb or pbtxt is
        stored. The SavedModel must only have one input per signature and only
        one output per signature.
      activation_type: one of the keys in saved_model.signature_def.keys().
      batch_size: batch size to use for individual inference ops.
      use_tqdm: Whether to print progress using tqdm.

    Raises:
      ValueError: if activation_type is not the name of a signature_def in the
        SavedModel.
      ValueError: if SavedModel.signature_def[activation_type] has an input
        other than 'sequence'.
      ValueError: if SavedModel.signature_def[activation_type] has more than
        one output.
    """
    self.batch_size = batch_size
    with tf.Graph().as_default() as graph:
      sess = tf.Session()
      saved_model = tf.saved_model.loader.load(
          sess, [tf.saved_model.tag_constants.SERVING], savedmodel_dir_path)

      if activation_type not in saved_model.signature_def.keys():
        raise ValueError('activation_type not in ' +
                         str(saved_model.signature_def.keys()) + '. Was ' +
                         activation_type + '.')

      signature = saved_model.signature_def[activation_type]

      expected_inputs = [
          protein_dataset.SEQUENCE_KEY, protein_dataset.SEQUENCE_LENGTH_KEY
      ]
      if set(signature.inputs.keys()) != set(expected_inputs):
        raise ValueError(
            'The signature_def does not have the expected inputs. Please '
            'reconfigure your saved model to only export signatures '
            'with sequence and length inputs. (Inputs were %s, expected %s)' %
            (str(signature.inputs.keys()), str(expected_inputs)))

      if len(signature.outputs.keys()) > 1:
        raise ValueError(
            'The signature_def given has more than one output. Please '
            'reconfigure your saved model to only export signatures '
            'with one output. (Outputs were %s)' %
            str(signature.outputs.keys()))
      fetch_tensor_name = list(signature.outputs.values())[0].name

    self._signature = signature
    self._graph = graph
    self._sess = sess
    self._fetch_tensor_name = fetch_tensor_name

    self._savedmodel_dir_path = savedmodel_dir_path
    self._activation_type = activation_type
    self._use_tqdm = use_tqdm

  def __repr__(self):
    return ('{} with feed tensors savedmodel_dir_path {} and '
            'activation_type {}').format(
                type(self).__name__, self._savedmodel_dir_path,
                self._activation_type)

  def _get_activations_for_batch(self,
                                 list_of_seqs,
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
      list_of_seqs: list of strings, with characters that are amino acids.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      np.array of floats containing the value from fetch_op.
    """
    one_hots = [utils.residues_to_one_hot(seq) for seq in list_of_seqs]
    padded_one_hots = utils.make_padded_np_array(one_hots)
    tensor_name = custom_tensor_to_retrieve or self._fetch_tensor_name
    with self._graph.as_default():
      return self._sess.run(
          tensor_name, {
              self._signature.inputs[protein_dataset.SEQUENCE_KEY].name:
                  padded_one_hots,
              self._signature.inputs[protein_dataset.SEQUENCE_LENGTH_KEY].name:
                  np.array([len(s) for s in list_of_seqs])
          })

  def get_activations(self, list_of_seqs, custom_tensor_to_retrieve=None):
    """Gets activations where batching may be needed to avoid OOM.

    Inputs are strings of amino acids, outputs are activations from the network.

    Args:
      list_of_seqs: list of strings as input for inference.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      concatenated numpy array of activations with shape [num_of_seqs, ...]
    """
    # TODO(theosanderson): inference can be made dramatically faster by sorting
    # list of_seqs by length before inference (and presumably reversing the
    # sort process afterwards)

    if not isinstance(list_of_seqs, list):
      raise ValueError('seq_input must be a list of strings.')
    logging.info('Predicting for %d sequences', len(list_of_seqs))

    if list_of_seqs == []:  # pylint: disable=g-explicit-bool-comparison
      return np.array([], dtype=float)

    batches = list(utils.batch_iterable(list_of_seqs, self.batch_size))
    itr = tqdm.tqdm(batches, position=0) if self._use_tqdm else batches
    output_matrix = None

    for i, batch in enumerate(itr):
      batch_activations = self._get_activations_for_batch(
          batch, custom_tensor_to_retrieve=custom_tensor_to_retrieve)

      if output_matrix is None:
        # Allocate matrix to store all activations:
        output_shape = list(batch_activations.shape)
        output_shape[0] = len(list_of_seqs)
        output_matrix = np.zeros(output_shape, np.float16)
      starting_index = i * self.batch_size
      output_matrix[starting_index:starting_index +
                    batch_activations.shape[0]] = batch_activations

    return output_matrix

  def get_variable(self, variable_name):
    """Gets the value of a variable from the graph.

    Args:
      variable_name: string name for retrieval. E.g. "vocab_name:0"

    Returns:
      output from TensorFlow from attempt to retrieve this value.
    """
    with self._graph.as_default():
      return self._sess.run(variable_name)


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
