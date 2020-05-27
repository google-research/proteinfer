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

"""Tests for module model_performance_analysis.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import parenthood_lib


class ParenthoodLibTest(parameterized.TestCase):

  def test_reverse_map_filters_items(self):
    test_parents = {'b': ['a', 'b'], 'a': ['a'], 'c': ['c'], 'd': ['a', 'b']}
    test_vocab = ['a', 'b', 'c']
    rev = parenthood_lib.reverse_map(test_parents, label_vocab=test_vocab)
    rev_map = {'a': {'a', 'b'}, 'b': {'b'}, 'c': {'c'}}
    self.assertEqual(rev, rev_map)

  def test_is_implied_by_something_else_positive_case(self):
    input_label = 'CL0192'
    input_reversed_normalizer = {
        'CL0192': {'PF00001', 'PF00002'},
        'PF00002': {'PF00002'},
    }
    input_other_labels_for_protein = {'CL0192', 'PF00002'}
    actual = parenthood_lib.is_implied_by_something_else(
        input_label, input_reversed_normalizer, input_other_labels_for_protein)
    expected = True
    self.assertEqual(actual, expected)

  def test_is_implied_by_something_else_negative_case(self):
    input_label = 'CL0192'
    input_reversed_normalizer = {
        'CL0192': {'PF00001', 'PF00002'},
    }
    input_other_labels_for_protein = {'CL0192'}
    actual = parenthood_lib.is_implied_by_something_else(
        input_label, input_reversed_normalizer, input_other_labels_for_protein)
    expected = False
    self.assertEqual(actual, expected)

  def test_filter_labels_to_most_specific(self):
    input_df = pd.DataFrame({
        'predicted_label': [
            frozenset(['CL0192', 'PF00002']),
            frozenset(['CL0192'])
        ]
    })
    input_normalizer = {
        'PF00002': frozenset(['PF00002', 'CL0192']),
        'CL0192': frozenset(['CL0192'])
    }

    actual = parenthood_lib.filter_labels_to_most_specific(
        input_df, input_normalizer)
    actual_predicted_labels = actual.predicted_label.values.tolist()

    expected = [frozenset(['PF00002']), frozenset(['CL0192'])]

    self.assertListEqual(actual_predicted_labels, expected)


if __name__ == '__main__':
  absltest.main()
