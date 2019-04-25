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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import protein_dataset
import utils
import tensorflow as tf

FLAGS = flags.FLAGS


def _numpy_one_hot(x, depth):
  """Convert numpy array of indexes into a full one-hot.

  Args:
    x: np.array.
    depth: int. maximum index in array (depth of one-hot output).

  Returns:
    np.array.
  """
  return np.eye(depth)[x]


def _dataset_iterator_to_list(itr, session):
  """Convert tf.data.Dataset iterator to a python list.

  Args:
    itr: tf.data.Dataset iterator.
    session: tf.Session.

  Returns:
    list.
  """
  actual_examples = []
  while True:
    try:
      actual_examples.append(session.run(itr.get_next()))
    except tf.errors.OutOfRangeError:
      break
  return actual_examples


class ProteinDatasetTest(parameterized.TestCase):

  def test_non_padded_dataset(self):
    # Set up test data.
    test_data_directory = os.path.join(
        FLAGS.test_srcdir,
        './testdata'
    )

    label_vocab_array = [
        'EMBL:AE017224', 'RefSeq:WP_002966386.1', 'ProteinModelPortal:P0CB34',
        'SMR:P0CB34', 'EnsemblBacteria:AAX75635', 'GeneID:29595679',
        'KEGG:bmb:BruAb2_0191', 'HOGENOM:HOG000133897', 'KO:K04078',
        'OMA:PGRIDDN', 'Proteomes:UP000000540', 'GO:GO:0005737',
        'GO:GO:0005524', 'GO:GO:0006457', 'CDD:cd00320', 'Gene3D:2.30.33.40',
        'HAMAP:MF_00580', 'InterPro:IPR020818', 'InterPro:IPR037124',
        'InterPro:IPR018369', 'InterPro:IPR011032', 'PANTHER:PTHR10772',
        'Pfam:PF00166', 'PRINTS:PR00297', 'SMART:SM00883', 'SUPFAM:SSF50129',
        'PROSITE:PS00681'
    ]
    with tf.Graph().as_default():
      sess = tf.Session()
      dataset = protein_dataset.non_batched_dataset(
          # Dev fold instead of train fold because the train fold is repeated.
          train_dev_or_test=protein_dataset.DEV_FOLD,
          label_vocab=label_vocab_array,
          data_root_dir=test_data_directory)
      example_itr = dataset.make_initializable_iterator()

      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      sess.run(example_itr.initializer)

    # Compute actual output
    actual_examples = _dataset_iterator_to_list(example_itr, sess)

    expected_length = 4

    # Compute expected values
    expected_sequence = 'MADIKFRPLHDRVVVRRVESEAKTAGGIIIPDTAKEKPQEGEVVAAGAGARDEAGKLVPLDVKAGDRVLFGKWSGTEVKIGGEDLLIMKESDILGIVG'
    expected_sequence_indexes = [
        utils.AMINO_ACID_VOCABULARY.index(x) for x in expected_sequence
    ]
    expected_sequence_one_hot = _numpy_one_hot(
        expected_sequence_indexes, depth=len(utils.AMINO_ACID_VOCABULARY))
    # Because the label vocab is exactly the labels in the first example, we
    # just get range(len(label_vocab_array))
    expected_label_indexes = range(len(label_vocab_array))
    expected_id = b'P0CB34'

    # Assert values correct
    self.assertLen(actual_examples, expected_length)
    np.testing.assert_equal(actual_examples[0][protein_dataset.SEQUENCE_KEY],
                            expected_sequence_one_hot)
    np.testing.assert_equal(
        actual_examples[0][protein_dataset.SEQUENCE_LENGTH_KEY],
        len(expected_sequence))
    np.testing.assert_equal(actual_examples[0][protein_dataset.LABEL_KEY],
                            expected_label_indexes)
    np.testing.assert_equal(actual_examples[0][protein_dataset.SEQUENCE_ID_KEY],
                            expected_id)

  def test_padded_dataset(self):
    # Set up test data.
    test_data_directory = os.path.join(
        FLAGS.test_srcdir,
        './testdata'
    )

    label_vocab_array = ['EMBL:AE017224']

    batch_size = 3

    with tf.Graph().as_default():
      sess = tf.Session()
      non_padded_dataset = protein_dataset.non_batched_dataset(
          # Dev fold instead of train fold because the train fold is repeated.
          train_dev_or_test=protein_dataset.DEV_FOLD,
          label_vocab=label_vocab_array,
          data_root_dir=test_data_directory)
      batched_dataset = protein_dataset.batched_dataset(
          non_padded_dataset, batch_size=batch_size, bucket_boundaries=[11000])
      batch_itr = batched_dataset.make_initializable_iterator()

      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      sess.run(batch_itr.initializer)

    # Compute actual output
    actual_examples = _dataset_iterator_to_list(batch_itr, sess)

    # Examine correctness of first element.
    actual_sequence_batch_shape = actual_examples[0][
        protein_dataset.SEQUENCE_KEY].shape
    expected_longest_sequence_len_in_first_batch = 98
    expected_first_batch_sequence_shape = (
        batch_size, expected_longest_sequence_len_in_first_batch,
        len(utils.AMINO_ACID_VOCABULARY))
    self.assertEqual(actual_sequence_batch_shape,
                     expected_first_batch_sequence_shape)

    actual_label_batch_shape = actual_examples[0][
        protein_dataset.LABEL_KEY].shape
    # Because the label vocab contains the labels in the first example, we
    # get len(label_vocab_array) as the number of labels.
    expected_batch_label_shape = (batch_size, len(label_vocab_array))
    self.assertEqual(actual_label_batch_shape, expected_batch_label_shape)

  def test_yield_examples(self):
    path = os.path.join(
        FLAGS.test_srcdir,
        './testdata/train*.tfrecord'
    )
    actual_examples = list(protein_dataset.yield_examples(path))
    expected_length = 4
    self.assertLen(actual_examples, expected_length)
    self.assertEqual(
        set(actual_examples[0].keys()),
        set([
            protein_dataset.SEQUENCE_KEY, protein_dataset.SEQUENCE_ID_KEY,
            protein_dataset.LABEL_KEY
        ]))


if __name__ == '__main__':
  tf.test.main()
