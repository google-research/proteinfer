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

"""Tests for proteinfer binary."""

from absl.testing import parameterized
import pandas as pd
import proteinfer
import test_util
import utils
import tensorflow.compat.v1 as tf


class ProteinferTest(parameterized.TestCase):

  def test_gcs_path_to_relative_unzipped_path(self):
    actual = proteinfer._gcs_path_to_relative_unzipped_path(
        utils.OSS_GO_ZIPPED_MODELS_URLS[0])
    expected = 'noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13703706'
    self.assertEqual(actual, expected)

  def test_parse_input(self):
    input_file_path = self.create_tempfile(content='>SEQUENCE_NAME\nACDE')
    input_text = proteinfer.parse_input_to_text(input_file_path.full_path)
    actual_df = proteinfer.input_text_to_df(input_text)
    expected = pd.DataFrame({
        'sequence_name': ['SEQUENCE_NAME'],
        'sequence': ['ACDE'],
    })

    # BioPython parses sequences as Bio.Seq.Seq which can, in most cases,
    # act as sequences, but in others can lead to surprising behavior. Ensure
    # we actually have a str.
    self.assertEqual(type(actual_df.sequence.values[0]), str)
    test_util.assert_dataframes_equal(self, actual_df, expected)

  def test_parse_input_malformed_fasta(self):
    # input is missing fasta header line marker.
    input_file_path = self.create_tempfile(content='SEQUENCE_NAME\nACDE')
    with self.assertRaisesRegex(ValueError, 'Failed to parse'):
      proteinfer.parse_input_to_text(input_file_path.full_path)

  def test_format_output_adds_description_and_formats_float_confidence(self):
    input_df = pd.DataFrame({
        'sequence_name': ['SEQ_A'],
        'predicted_label': ['Pfam:PF000042'],
        'confidence': [.991]
    })
    label_to_description = {'Pfam:PF000042': 'Oxygen carrier'}
    num_decimal_places = 2

    actual = proteinfer.format_df_for_output(input_df, label_to_description,
                                             num_decimal_places)
    expected = pd.DataFrame({
        'sequence_name': ['SEQ_A'],
        'predicted_label': ['Pfam:PF000042'],
        'confidence': [.99],
        'description': ['Oxygen carrier']
    })

    test_util.assert_dataframes_equal(self, actual, expected)

  @parameterized.parameters(
      dict(input_confidence=1., num_decimal_places=2, expected=1.0),
      dict(input_confidence=1., num_decimal_places=3, expected=1.0),
      dict(input_confidence=0.1, num_decimal_places=2, expected=0.1),
      dict(input_confidence=0.01, num_decimal_places=2, expected=0.01),
      dict(input_confidence=0.006, num_decimal_places=2, expected=0.01),
      dict(input_confidence=0.001, num_decimal_places=3, expected=0.001),
  )
  def test_format_float_confidence(self, input_confidence, num_decimal_places,
                                   expected):
    actual = proteinfer._format_float_confidence_for_output(
        input_confidence, num_decimal_places)
    self.assertEqual(actual, expected)

  def test_load_models_raises_on_model_missing_no_ensemble(self):
    expected_err_contents = ('Unable to find cached models in FAKE_PATH. Make '
                             'sure you have installed the models by running\n'
                             '    install_models.py '
                             '--model_cache_path=FAKE_PATH\nThen try rerunning '
                             'this script.')
    with self.assertRaises(ValueError) as exc:
      proteinfer.load_models(
          model_cache_path='FAKE_PATH', num_ensemble_elements=1)

    actual = exc.exception.args[0]

    self.assertIn(expected_err_contents, actual)

  def test_load_models_raises_on_model_missing_with_ensemble(self):
    expected_err_contents = ('Unable to find cached models in FAKE_PATH. Make '
                             'sure you have installed the entire ensemble of '
                             'models by running\n    install_models.py '
                             '--install_ensemble '
                             '--model_cache_path=FAKE_PATH\nThen try rerunning '
                             'this script.')
    with self.assertRaises(ValueError) as exc:
      proteinfer.load_models(
          model_cache_path='FAKE_PATH', num_ensemble_elements=3)

    actual = exc.exception.args[0]
    self.assertIn(expected_err_contents, actual)

  @parameterized.named_parameters(
      dict(
          testcase_name='orders by sequence name\'s original ordering',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_B', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'Pfam:PF00002'],
              'confidence': [.1, .9],
              'description': ['First GPCRA', 'Second GPCR'],
          }),
          expected=pd.DataFrame({
              # Note that this df is not sorted by the sequence name's column
              # alphabetically; instead it preserves the original ordering.
              'sequence_name': ['SEQ_B', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'Pfam:PF00002'],
              'confidence': [.1, .9],
              'description': ['First GPCRA', 'Second GPCR'],
          }),
      ),
      dict(
          testcase_name='orders by confidences given same label class',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'Pfam:PF00002'],
              'confidence': [.1, .9],
              'description': ['First GPCRA', 'Second GPCR'],
          }),
          expected=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00002', 'Pfam:PF00001'],
              'confidence': [.9, .1],
              'description': ['Second GPCR', 'First GPCRA'],
          }),
      ),
      dict(
          testcase_name='orders Pfam then EC then GO',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A', 'SEQ_A'],
              'predicted_label': ['GO:0123456', 'EC:1.2.3.-', 'Pfam:PF00001'],
              'confidence': [.1, .9, .3],
              'description': ['go label', 'ec label', 'pfam label'],
          }),
          expected=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'EC:1.2.3.-', 'GO:0123456'],
              'confidence': [.3, .9, .1],
              'description': ['pfam label', 'ec label', 'go label'],
          }),
      ),
      dict(
          testcase_name='For EC, orders by label alphabetically, not description alphabetically (despite confidences)',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['EC:1.2.3.-', 'EC:1.2.-.-'],
              'confidence': [1., .9],
              'description': [
                  'AAA alphabetically FIRST EC label',
                  'ZZZ alphabetically LAST EC label'
              ],
          }),
          expected=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['EC:1.2.-.-', 'EC:1.2.3.-'],
              'confidence': [.9, 1.],
              'description': [
                  'ZZZ alphabetically LAST EC label',
                  'AAA alphabetically FIRST EC label'
              ],
          }),
      ),
      dict(
          testcase_name='orders by description given same label class and confidence',
          input_df=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00001', 'Pfam:PF00002'],
              'confidence': [1., 1.],
              'description': ['ZZZZ', 'AAAA'],
          }),
          expected=pd.DataFrame({
              'sequence_name': ['SEQ_A', 'SEQ_A'],
              'predicted_label': ['Pfam:PF00002', 'Pfam:PF00001'],
              'confidence': [1., 1.],
              'description': ['AAAA', 'ZZZZ'],
          }),
      ),
  )
  def test_order_df_for_output(self, input_df, expected):
    actual = proteinfer.order_df_for_output(input_df)
    test_util.assert_dataframes_equal(self, actual, expected)


if __name__ == '__main__':
  tf.test.main()
