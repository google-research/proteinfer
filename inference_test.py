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

"""Tests for module inference_lib.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import inference

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
    with self.assertRaisesRegexp(ValueError, 'must be a list of strings'):
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


if __name__ == '__main__':
  absltest.main()
