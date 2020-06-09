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
import os



from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import inference
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class _InferrerFixture(object):
  activation_type = 'serving_default'

  def get_activations(self, l):
    return np.array([len(s) for s in l])


class InferenceLibTest(parameterized.TestCase):

  def setUp(self):
    super(InferenceLibTest, self).setUp()
    initial = './'
    self.saved_model_path = os.path.join(FLAGS.test_srcdir,
                                         (initial + 'testdata/saved_model/'))

  def testBatchedInference(self):
    inferrer = inference.Inferrer(self.saved_model_path, batch_size=5)

    input_seq = 'AP'
    for total_size in range(15):
      full_list = [input_seq] * total_size
      activations = inferrer.get_activations(full_list)
      self.assertLen(full_list, activations.shape[0])

  def testStringInput(self):
    inferrer = inference.Inferrer(self.saved_model_path)
    # Simulate failure to use a list.
    with self.assertRaisesRegex(ValueError, 'must be a list of strings'):
      inferrer.get_activations('QP')

  def testGetVariable(self):
    inferrer = inference.Inferrer(self.saved_model_path)
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


if __name__ == '__main__':
  absltest.main()
