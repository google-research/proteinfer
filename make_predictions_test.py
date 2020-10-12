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
import make_predictions


class MakePredictionsTest(parameterized.TestCase):

  def test_generate_predictions(self):
      make_predictions.generate_predictions( "testdata/test*.tfrecord","./output_test.file",  "testdata/saved_model/" )


if __name__ == '__main__':
    absltest.main()
