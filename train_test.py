# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for train.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile

from absl import flags
from absl.testing import parameterized
import numpy as np
import hparams_sets
import protein_dataset
import train
import tensorflow as tf

FLAGS = flags.FLAGS


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self._test_data_directory = os.path.join(
        FLAGS.test_srcdir,
        './testdata'
    )

  def test_parse_label_vocab(self):
    label_vocab_path = os.path.join(self._test_data_directory,
                                    'label_vocab.tsv')
    actual = train.parse_label_vocab(label_vocab_path)
    expected = np.array([
        'GO:GO:0005737', 'GO:GO:0005524', 'GO:GO:0006457', 'GO:GO:0020038',
        'GO:GO:0020039', 'GO:GO:0020040', 'GO:GO:0020041', 'GO:GO:0020042',
        'GO:GO:0020043', 'GO:GO:0020044', 'GO:GO:0020045'
    ])
    np.testing.assert_array_equal(actual, expected)

  def test_train_gives_non_nan_loss(self):
    output_dir = tempfile.mkdtemp('test_model_output')

    evaluation_results, export_results = train.train(
        data_base_path=self._test_data_directory,
        output_dir=output_dir,
        label_vocab_path=os.path.join(self._test_data_directory,
                                      'label_vocab.tsv'),
        hparams_set_name=hparams_sets.small_test_model.__name__,
        train_fold=protein_dataset.TRAIN_FOLD,
        eval_fold=protein_dataset.TEST_FOLD)

    self.assertTrue(np.isfinite(evaluation_results['loss']))

    saved_model_path = export_results[0]
    # Check we can load the saved_model without exceptions:
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 saved_model_path)
      label_vocab = tf.get_default_graph().get_tensor_by_name('label_vocab:0')
      self.assertLen(label_vocab.shape, 1)
      decision_threshold = sess.run(
          tf.get_default_graph().get_tensor_by_name('decision_threshold:0'))
      self.assertGreater(decision_threshold, 0)
      self.assertLess(decision_threshold, 1)


if __name__ == '__main__':
  tf.test.main()
