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
"""Tests for module colab_evaluation.py."""

import gzip
import os
import time
import math

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import colab_evaluation
import inference
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class ColabEvaluationTest(parameterized.TestCase):
    def _generate_random_inferences(self, n):
        serialized_inferences = []
        accessions_list = []
        activations_list = []

        for _ in range(n):
            accession = f"ACCESSION_{time.time()}"
            activations = np.random.rand(100)
            accessions_list.append(accession)
            activations_list.append(activations)
            serialized_inferences.append(
                inference.serialize_inference_result(accession, activations))

        return serialized_inferences, accessions_list, activations_list

    @parameterized.parameters([{'batch_size': 1}, {'batch_size': 9}])
    def test_batched_inferences_from_dir(self, batch_size, num_examples=100):

        # Create input inference results.

        serialized_inferences, accessions_list, activations_list = self._generate_random_inferences(
            num_examples)

        shard_1_contents = b"\n".join(serialized_inferences[0:60])
        shard_2_contents = b"\n".join(serialized_inferences[60:])

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

        # Read these shards.

        iterator = colab_evaluation.batched_inferences_from_dir(
            shard_dir.full_path, batch_size=batch_size)

        actual = list(iterator)

        # Check output.

        self.assertEqual(len(actual), math.ceil(num_examples / batch_size))
        self.assertEqual(actual[0][0][0], accessions_list[0])
        if batch_size > 1:
            self.assertEqual(actual[1][0][1], accessions_list[batch_size + 1])
        np.testing.assert_equal(actual[0][1][0], activations_list[0])
        if batch_size > 1:
            np.testing.assert_equal(actual[1][1][1],
                                    activations_list[batch_size + 1])

    def test_make_tidy_df_from_seq_names_and_prediction_array(self):
        vocab = ["ENTRY0", "ENTRY1", "ENTRY2"]
        sequence_names = ['SEQ0', 'SEQ1']
        predictions_array = np.array([[0.1, 0.9, 0.5], [1, 1, 1]])
        min_decision_threshold = 0.4
        actual_df = colab_evaluation._make_tidy_df_from_seq_names_and_prediction_array(
            sequence_names,
            predictions_array,
            vocab,
            min_decision_threshold=min_decision_threshold)
        expected_df = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ0', 'SEQ1', 'SEQ1', 'SEQ1'],
            'label': ['ENTRY1', 'ENTRY2', 'ENTRY0', 'ENTRY1', 'ENTRY2'],
            'value': [0.9, 0.5, 1.0, 1.0, 1.0]
        })
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_make_tidy_df_from_ground_truth(self):
        input_df = pd.DataFrame({
            'sequence_name': ['SEQ0', 'SEQ1', 'SEQ2', 'SEQ3'],
            'true_label': [['ENTRY1'], ['ENTRY1', 'ENTRY2'], [], ['ENTRY6']]
        })
        actual_df = colab_evaluation.make_tidy_df_from_ground_truth(input_df)
        expected_df = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ1', 'SEQ1', 'SEQ3'],
            'label': ['ENTRY1', 'ENTRY1', 'ENTRY2', 'ENTRY6'],
            'gt': [True, True, True, True]
        })
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_merge_predictions_and_ground_truth(self):
        pred = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ0', 'SEQ1', 'SEQ1', 'SEQ1'],
            'label': ['ENTRY1', 'ENTRY2', 'ENTRY0', 'ENTRY1', 'ENTRY2'],
            'value': [0.9, 0.5, 1.0, 1.0, 1.0]
        })
        gt = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ1', 'SEQ1', 'SEQ3'],
            'label': ['ENTRY1', 'ENTRY1', 'ENTRY2', 'ENTRY6'],
            'gt': [True, True, True, True]
        })
        actual_df = colab_evaluation.merge_predictions_and_ground_truth(
            pred, gt)
        expected_df = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ0', 'SEQ1', 'SEQ1', 'SEQ1', 'SEQ3'],
            'label':
            ['ENTRY1', 'ENTRY2', 'ENTRY0', 'ENTRY1', 'ENTRY2', 'ENTRY6'],
            'value': [0.9, 0.5, 1.0, 1.0, 1.0, False],
            'gt': [True, False, False, True, True, True]
        })
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_get_pr_curve_df(self):
        pred = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ0', 'SEQ1', 'SEQ1', 'SEQ1'],
            'label': ['ENTRY1', 'ENTRY2', 'ENTRY0', 'ENTRY1', 'ENTRY2'],
            'value': [0.9, 0.5, 1.0, 1.0, 1.0]
        })
        gt = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ1', 'SEQ1', 'SEQ3'],
            'label': ['ENTRY1', 'ENTRY1', 'ENTRY2', 'ENTRY6'],
            'gt': [True, True, True, True]
        })
        pr_curve = colab_evaluation.get_pr_curve_df(pred, gt)

        np.testing.assert_almost_equal(pr_curve['recall'],
                                       np.array([1, 0.75, 0.75, .5]))
        np.testing.assert_almost_equal(
            pr_curve['precision'], np.array([0.6666667, 0.6, 0.75, 0.6666667]))
        np.testing.assert_almost_equal(
            pr_curve['f1'], np.array([0.8, 0.6666667, 0.75, 0.5714286]))

    def test_assign_tp_fp_fn(self):
        pred = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ0', 'SEQ1', 'SEQ1', 'SEQ1'],
            'label': ['ENTRY1', 'ENTRY2', 'ENTRY0', 'ENTRY1', 'ENTRY2'],
            'value': [0.9, 0.5, 1.0, 1.0, 1.0]
        })
        gt = pd.DataFrame({
            'up_id': ['SEQ0', 'SEQ1', 'SEQ1', 'SEQ3'],
            'label': ['ENTRY1', 'ENTRY1', 'ENTRY2', 'ENTRY6'],
            'gt': [True, True, True, True]
        })
        tp_fp_fn = colab_evaluation.assign_tp_fp_fn(pred, gt, threshold=0.5)

        expected = pd.DataFrame({
            'tp': [True, False, False, True, True, False],
            'fp': [False, False, True, False, False, False],
            'fn': [False, False, False, False, False, True]
        })
        actual = tp_fp_fn.loc[:, ["tp", "fp", "fn"]]
        pd.testing.assert_frame_equal(expected, actual)


if __name__ == '__main__':
    absltest.main()
