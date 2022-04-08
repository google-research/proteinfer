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

"""Tests for module inference.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip



from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import scipy.sparse
import inference
import test_util
import utils
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class _InferrerFixture(object):
  """A mock inferrer object.

  See docstring for get_activations.
  """
  activation_type = 'serving_default'

  def __init__(self, activation_rank=1):
    """Constructs a mock inferrer with activation output of specified rank.

    Args:
      activation_rank: int. Use 1 for activations that have a single float per
        sequence, 2. for a vector per sequence, etc.
    """
    self._activation_rank = activation_rank

  def get_variable(self, x):
    if x == 'label_vocab:0':
      return np.array(['LABEL1'])
    else:
      raise ValueError(
          'Fixture does not have an implementation for this variable')

  def get_activations(self, input_seqs):
    """Returns a np.array with contents that are the length of each seq.

    The shape of the np.array is dictated by self._activation_rank - see
    docstring of __init__ for more information.

    Args:
      input_seqs: list of string.

    Returns:
      np.array of rank self._activation_rank, where the entries are the length
      of each input seq. See Inferrer.get_activations for more information 
      about what this class is mocking.
    """
    dense = np.reshape([len(s) for s in input_seqs],
                      [-1] + [1] * (self._activation_rank - 1))
    return np.array([scipy.sparse.coo_matrix(x) for x in dense])


class InGraphInferrerTest(tf.test.TestCase, parameterized.TestCase):

  def testCanInfer(self):

    graph = tf.Graph()
    with graph.as_default():
      sequences = tf.placeholder(shape=[None], dtype=tf.string)
      output_tensor = inference.in_graph_inferrer(
          sequences, test_util.savedmodel_path(),
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    input_seqs = [''.join(utils.FULL_RESIDUE_VOCAB), 'ACD']
    with self.session(graph=graph) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      result = sess.run(output_tensor, feed_dict={sequences: input_seqs})

    self.assertLen(result, 2)


class InferenceLibTest(parameterized.TestCase, tf.test.TestCase):

  def testBatchedInference(self):
    inferrer = inference.Inferrer(test_util.savedmodel_path(), batch_size=5)

    input_seq = 'AP'
    for total_size in range(15):
      full_list = [input_seq] * total_size
      activations = inferrer.get_activations(full_list)
      self.assertLen(full_list, activations.shape[0])

  def testSortUnsortInference(self):
    inferrer = inference.Inferrer(test_util.savedmodel_path(), batch_size=1)

    input_seqs = ['AP', 'APP', 'AP']
    # Sorting will move long sequence to the end.
    activations = inferrer.get_activations(input_seqs)
    # Make sure it gets moved back to the middle.
    self.assertAllClose(activations[0].todense(), activations[2].todense())
    self.assertNotAllClose(activations[0].todense(), activations[1].todense())

  def testStringInput(self):
    inferrer = inference.Inferrer(test_util.savedmodel_path())
    # Simulate failure to use a list.
    with self.assertRaisesRegex(
        ValueError, '`list_of_seqs` should be convertible to a '
        'numpy vector of strings. Got *'):
      inferrer.get_activations('QP')

  def testMemoizedInferrerLoading(self):
    inferrer = inference.memoized_inferrer(
        test_util.savedmodel_path(), memoize_inference_results=True)
    memoized_inferrer = inference.memoized_inferrer(
        test_util.savedmodel_path(), memoize_inference_results=True)

    self.assertIs(inferrer, memoized_inferrer)

  def testMemoizedInferenceResults(self):
    inferrer = inference.Inferrer(
        test_util.savedmodel_path(), memoize_inference_results=True)
    activations = inferrer._get_activations_for_batch(('ADE',))
    memoized_activations = inferrer._get_activations_for_batch(('ADE',))

    self.assertIs(activations, memoized_activations)

  def testGetVariable(self):
    inferrer = inference.Inferrer(test_util.savedmodel_path())
    output = inferrer.get_variable('conv1d/bias:0')
    self.assertNotEmpty(output)

  def test_predictions_for_df(self):
    inferrer_fixture = _InferrerFixture()
    input_seqs = ['AAAA', 'DDD', 'EE', 'W']
    input_df = pd.DataFrame({
        'sequence_name': input_seqs,
        'sequence': input_seqs
    })
    actual_output_df = inference.predictions_for_df(input_df, inferrer_fixture)

    self.assertEqual(actual_output_df['predictions'].values.tolist(),
                     [4, 3, 2, 1])

    self.assertEqual(actual_output_df.sequence_name.values.tolist(), input_seqs)

  def test_serialize_deserialize_inference_result(self):
    input_accession = 'ACCESSION'
    input_activations = np.array([1., 2., 3.])

    serialized = inference.serialize_inference_result(input_accession,
                                                      input_activations)
    deserialized_actual_accession, deserialized_actual_activations = inference.deserialize_inference_result(
        serialized)

    self.assertEqual(deserialized_actual_accession, input_accession)
    np.testing.assert_array_equal(deserialized_actual_activations,
                                  input_activations)

  def test_parse_sharded_inference_results(self):
    # Create input inference results.
    input_accession_1 = 'ACCESSION_1'
    input_activations_1 = np.array([1., 2., 3.])

    input_accession_2 = 'ACCESSION_2'
    input_activations_2 = np.array([4., 5., 6.])

    input_accession_3 = 'ACCESSION_3'
    input_activations_3 = np.array([7., 8., 9.])

    # Create files and a directory containing those inference results.
    shard_1_contents = inference.serialize_inference_result(
        input_accession_1,
        input_activations_1) + b'\n' + inference.serialize_inference_result(
            input_accession_2, input_activations_2)

    shard_2_contents = inference.serialize_inference_result(
        input_accession_3, input_activations_3)

    shard_dir = self.create_tempdir()

    shard_1_filename = shard_dir.create_file('shard_1').full_path
    shard_2_filename = shard_dir.create_file('shard_2').full_path

    # Write contents to a gzipped file.
    with tf.io.gfile.GFile(shard_1_filename, 'wb') as f:
      with gzip.GzipFile(fileobj=f, mode='wb') as f_gz:
        f_gz.write(shard_1_contents)

    with tf.io.gfile.GFile(shard_2_filename, 'wb') as f:
      with gzip.GzipFile(fileobj=f, mode='wb') as f_gz:
        f_gz.write(shard_2_contents)

    actual = inference.parse_all_shards(shard_dir.full_path).values
    actual = sorted(actual, key=lambda x: x[0])

    self.assertEqual(actual[0][0], input_accession_1)
    self.assertEqual(actual[1][0], input_accession_2)
    self.assertEqual(actual[2][0], input_accession_3)

    np.testing.assert_array_equal(actual[0][1], input_activations_1)
    np.testing.assert_array_equal(actual[1][1], input_activations_2)
    np.testing.assert_array_equal(actual[2][1], input_activations_3)

  @parameterized.named_parameters(
      dict(
          testcase_name='filters one sequence',
          input_df=pd.DataFrame({
              'sequence_name': ['seq1', 'seq2'],
              'sequence': ['ACDE', 'WWWYYY']
          }),
          threshold=5.,
          expected=pd.DataFrame({
              'sequence_name': ['seq2'],
              'confidence': [6.],
              'predicted_label': ['LABEL1'],
          })),
      dict(
          testcase_name='filters no sequences, but preserves input sequence_name ordering',
          input_df=pd.DataFrame({
              'sequence_name': ['seq2', 'seq1'],
              'sequence': ['WWWYYY', 'ACDE']
          }),
          threshold=2.,
          expected=pd.DataFrame({
              # Note: doesn't sort by sequence_name.
              'sequence_name': ['seq2', 'seq1'],
              'confidence': [6., 4.],
              'predicted_label': ['LABEL1', 'LABEL1'],
          })),
  )
  def testGetPredsAboveThreshold(self, input_df, expected, threshold):
    inferrer_list = [_InferrerFixture(activation_rank=2)]

    # Assert that the first sequence was removed.
    actual = inference.get_preds_at_or_above_threshold(input_df, inferrer_list,
                                                       threshold)
    test_util.assert_dataframes_equal(self, actual, expected)

  def testGetPredsAboveThresholdRaisesOnZeroThreshold(self):
    inferrer_list = []
    input_df = pd.DataFrame()

    with self.assertRaisesRegex(ValueError, '0'):
      inference.get_preds_at_or_above_threshold(input_df, inferrer_list, 0.)


if __name__ == '__main__':
  absltest.main()
