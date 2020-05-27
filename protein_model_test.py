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

"""Tests for protein_model.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
import protein_model
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class ProteinModelTest(parameterized.TestCase):

  def testF1Score(self):
    """Tests the F1 score metric."""
    labels = tf.constant([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=tf.int32)
    # Sensitivity: 1/2
    # Specificity: 1/2
    # F1-score: 1/2
    predictions = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]],
                              dtype=tf.float32)
    f1, update_op = protein_model._f1_score(labels, predictions)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertAlmostEqual(1. / 2, f1.eval(), places=4)

  def testMeanExampleWiseF1Score(self):
    """Tests the F1 score metric."""
    labels = tf.constant([[1, 0], [0, 1], [0, 1], [1, 0], [0, 0]],
                         dtype=tf.int32)
    # Sensitivity: 1/2
    # Specificity: 1/2
    # F1-score: 1/2
    predictions = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1]],
                              dtype=tf.float32)
    f1, update_op = protein_model._mean_examplewise_f1_score(
        labels, predictions)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(update_op)
      self.assertAlmostEqual(1. / 2, f1.eval(), places=4)

  def testRecallAtK(self):
    labels = tf.convert_to_tensor([[1, 0, 0], [0, 1, 1], [0, 0, 0]],
                                  dtype=tf.float32)
    predictions = tf.convert_to_tensor([[0.5, 1.0, 0.0], [0.5, 0.0, 1.0],
                                        [0.25, 0.25, 0.4]])

    k_values = [0, 1, 2]
    expected_recall_values_for_each_k = [0, 0.25, 0.75]
    # For k = 0 we get 0% from both of first two (3rd example is always NA)
    # For k = 1 we get 0% from first example and 50% from second
    # For k = 2 we get 100% from first example and 50% from second

    values_and_updates = [
        protein_model._custom_recall_at_k(
            labels_as_multi_hot=labels, predictions=predictions, k=k)
        for k in k_values
    ]

    with tf.Session() as sess:
      for i, value_and_update in enumerate(values_and_updates):
        value, update_op = value_and_update
        sess.run(tf.initialize_local_variables())
        sess.run(update_op)
        actual_recall = sess.run(value)
        self.assertEqual(actual_recall, expected_recall_values_for_each_k[i])

  @parameterized.named_parameters(
      dict(
          testcase_name='float values',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[2, 3],
          expected=[[[11], [21], [0]], [[41], [51], [61]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='no padding',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[3, 3],
          expected=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='all padding',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[0, 0],
          expected=[[[0.], [0.], [0.]], [[0.], [0.], [0.]]],
          sentinel=0.,
      ),
      dict(
          testcase_name='different sentinel',
          padded_representations=[[[11.], [21.], [31.]], [[41.], [51.], [61.]]],
          sequence_lengths=[0, 0],
          expected=[[[-99.], [-99.], [-99.]], [[-99.], [-99.], [-99.]]],
          sentinel=-99.,
      ),
      dict(
          testcase_name='embedding dimension size > 1',
          padded_representations=[[[11., -1.], [21., -2.], [31., -3.]],
                                  [[41., -4.], [51., -5.], [61., -6.]]],
          sequence_lengths=[2, 3],
          expected=[[[11., -1.], [21., -2.], [0., 0.]],
                    [[41., -4.], [51., -5.], [61., -6.]]],
          sentinel=0.,
      ),
  )
  def testSetPaddingToSentinel(self, padded_representations, sequence_lengths,
                               expected, sentinel):
    with tf.Graph().as_default():
      with tf.Session() as sess:
        padded_representations = tf.convert_to_tensor(padded_representations)
        sequence_lengths = tf.convert_to_tensor(sequence_lengths)
        actual = sess.run(
            protein_model._set_padding_to_sentinel(padded_representations,
                                                   sequence_lengths, sentinel))
        np.testing.assert_array_almost_equal(actual, expected)

  @parameterized.parameters(
      dict(
          input_array=[[0, 1], [2, 3]],
          vocab_size=4,
          expected=[[1, 1, 0, 0], [0, 0, 1, 1]]),
      dict(
          input_array=[[3, -1], [2, 3]],
          vocab_size=4,
          expected=[[0, 0, 0, 1], [0, 0, 1, 1]]))
  def testIndicesToMultiHot(self, input_array, vocab_size, expected):

    with tf.Graph().as_default():
      with tf.Session() as sess:
        input_array = tf.convert_to_tensor(input_array)
        actual = sess.run(
            protein_model._indices_to_multihot(input_array, vocab_size))
        np.testing.assert_array_almost_equal(actual, expected)


if __name__ == '__main__':
  tf.test.main()
