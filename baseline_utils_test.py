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

# Lint as: python2, python3
# pylint: disable=line-too-long
"""Tests for module model_performance_analysis.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

import baseline_utils
import test_util
import tensorflow.compat.v1 as tf


def _write_to_file(contents):
  tmpfile_name = tempfile.mktemp()
  with tf.io.gfile.GFile(tmpfile_name, "w") as f:
    f.write(contents.encode("utf-8"))
  return tmpfile_name


class BaselineUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="has fasta character >",
          header='>accession="ACCESSION"\tlabels="label1,label2"',
          expected="ACCESSION",
      ),
      dict(
          testcase_name="does not have character >",
          header='accession="ACCESSION"\tlabels="label1,label2"',
          expected="ACCESSION",
      ),
  )
  def test_get_sequence_name_from_sequence_header(self, header, expected):
    actual = baseline_utils._get_sequence_name_from_sequence_header(header)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="two labels",
          header='>accession="ACCESSION"\tlabels="label1,label2"',
          expected={"label1", "label2"},
      ),
      dict(
          testcase_name="zero labels",
          header='>accession="ACCESSION"\tlabels=""',
          expected=set(),
      ),
  )
  def test_get_labels_from_sequence_header(self, header, expected):
    actual = baseline_utils._get_labels_from_sequence_header(header)
    self.assertEqual(actual, expected)

  def test_load_ground_truth(self):
    input_fasta = ('>accession="ACCESSION"\tlabels="GO:101010,EC:9.9.9.9"\n'
                   "ADE\n"
                   '>accession="ACCESSION2"\tlabels="EC:1.2.-.-"\n'
                   "WWWW\n")
    tmpfile_name = _write_to_file(input_fasta)
    actual = baseline_utils.load_ground_truth(tmpfile_name)

    expected = pd.DataFrame({
        "sequence_name": ["ACCESSION", "ACCESSION2"],
        "true_label": [{"GO:101010", "EC:9.9.9.9"}, {"EC:1.2.-.-"}],
        "sequence": ["ADE", "WWWW"]
    })

    test_util.assert_dataframes_equal(self, actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="no inputs, one thing in vocab",
          input_row=pd.Series({
              "predicted_label": frozenset([]),
              "bit_score": 99.
          }),
          input_label_vocab=np.array(["PF00001"]),
          expected=[0.],
      ),
      dict(
          testcase_name="one input, one thing in vocab",
          input_row=pd.Series({
              "predicted_label": frozenset(["PF00001"]),
              "bit_score": 99.
          }),
          input_label_vocab=np.array(["PF00001"]),
          expected=[99.],
      ),
      dict(
          testcase_name="one input, two things in vocab",
          input_row=pd.Series({
              "predicted_label": frozenset(["PF00001"]),
              "bit_score": 99.
          }),
          input_label_vocab=np.array(["PF00001", "PF99999"]),
          expected=[99., 0.],
      ),
      dict(
          testcase_name="two inputs, two things in vocab",
          input_row=pd.Series({
              "predicted_label": frozenset(["PF00001", "PF99999"]),
              "bit_score": 99.
          }),
          input_label_vocab=np.array(["PF00001", "PF99999"]),
          expected=[99., 99.],
      ),
  )
  def test_blast_row_to_confidence_array(self, input_row, input_label_vocab,
                                         expected):
    lookup = {k: i for i, k in enumerate(input_label_vocab)}

    actual = baseline_utils._blast_row_to_confidence_array(
        input_row, input_label_vocab, lookup)
    np.testing.assert_allclose(actual, expected)

  def test_load_blast_output(self):
    input_test_fasta = (
        '>accession="ACCESSION"\tlabels="GO:101010,EC:9.9.9.9"\n'
        "ADE\n"
        '>accession="ACCESSION2"\tlabels="EC:1.2.-.-"\n'
        "WWWW\n")
    test_fasta_filename = _write_to_file(input_test_fasta)
    ground_truth_test = baseline_utils.load_ground_truth(test_fasta_filename)

    input_train_fasta = (
        '>accession="MATCHACCESSION"\tlabels="GO:101010,EC:9.9.9.9,Pfam:PF12345"\n'
        "ADE\n")
    train_fasta_filename = _write_to_file(input_train_fasta)
    ground_truth_train = baseline_utils.load_ground_truth(train_fasta_filename)

    # Missing second sequence in ground truth.
    input_blast = (
        'accession="ACCESSION"\taccession="MATCHACCESSION"\t82.456\t57\t10\t0\t1\t57\t1\t57\t6.92e-21\t79.3\n'
    )
    input_label_vocab = np.array(
        ["EC:1.2.-.-", "EC:9.9.9.9", "GO:101010", "Pfam:PF12345"])
    blast_filename = _write_to_file(input_blast)
    actual = baseline_utils.load_blast_output(
        filename=blast_filename,
        label_vocab=input_label_vocab,
        test_data_ground_truth=ground_truth_test,
        training_data_ground_truth=ground_truth_train)

    expected = pd.DataFrame({
        "sequence_name": ["ACCESSION", "ACCESSION2"],
        "closest_sequence": ["MATCHACCESSION", float("nan")],
        "true_label": [{"GO:101010", "EC:9.9.9.9"}, {"EC:1.2.-.-"}],
        "predicted_label": [{"GO:101010", "EC:9.9.9.9", "Pfam:PF12345"},
                            frozenset()],
        "percent_seq_identity": [82.456, float("nan")],
        "e_value": [6.92e-21, float("nan")],
        "bit_score": [79.3, 0.0],
    })

    test_util.assert_dataframes_equal(
        self,
        # Assert dataframes equal except for predictions column.
        # Rely on unit testing for predictions column instead to increase
        # test clarity. See test_blast_row_to_confidence_array above.
        actual.drop(columns=["predictions"]),
        expected,
        nan_equals_nan=True)

  def test_limit_set_of_labels(self):
    # Set up input data.
    input_df = pd.DataFrame(
        {"labels": [frozenset(["a"]), frozenset(["a", "b"])]})
    acceptable_labels = frozenset(["a"])
    column_to_limit = "labels"

    # Assert input dataframe was not modified later on, so save a copy.
    input_df_copy = input_df.copy()

    # Compute actual.
    actual = baseline_utils.limit_set_of_labels(input_df, acceptable_labels,
                                                column_to_limit)
    expected = pd.DataFrame({"labels": [frozenset(["a"]), frozenset(["a"])]})

    # Test assertions.
    test_util.assert_dataframes_equal(self, actual, expected)

    # Assert input dataframe was not modified.
    test_util.assert_dataframes_equal(self, input_df, input_df_copy)

  def test_limit_labels_for_label_normalizer(self):
    input_label_normalizer = {
        "a": ["a", "b", "c"],
        "DDDD": ["XXXX"],
        "b": ["YYYY", "b"]
    }
    input_acceptable_labels = frozenset(["a", "b"])

    actual = baseline_utils.limit_labels_for_label_normalizer(
        input_label_normalizer, input_acceptable_labels)
    expected = {"a": ["a", "b"], "b": ["b"]}

    self.assertDictEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="one sequence, one label row, no extra nonlabel entries",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	TIGRFAM	TIGR00506	ribB: 3,4-dihydroxy-2-butanone-4-phosphate synthase	13	209	1.5E-86	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={
              "B7UIV5": {"GO:0008686", "GO:0009231"}
          },
      ),
      dict(
          testcase_name="one sequence, one label row, extra nonlabel entries",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	TIGRFAM	TIGR00506	ribB: 3,4-dihydroxy-2-butanone-4-phosphate synthase	13	209	1.5E-86	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	Gene3D	G3DSA:3.90.870.10		1	217	5.1E-95T21-10-2019""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={"B7UIV5": {"GO:0008686", "GO:0009231"}},
      ),
      dict(
          testcase_name="one sequence, no labels",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	Gene3D	G3DSA:3.90.870.10		1	217	5.1E-95T21-10-2019""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={"B7UIV5": set()},
      ),
      dict(
          testcase_name="one sequence, multiple label rows, extra nonlabel entries",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	PANTHER	PTHR21327:SF38		1	217	8.2E-126	T21-10-2019
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	TIGRFAM	TIGR00506	ribB: 3,4-dihydroxy-2-butanone-4-phosphate synthase	13	209	1.5E-86	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	Hamap	MF_00180	3,4-dihydroxy-2-butanone 4-phosphate synthase [ribB].	11	213	43.238	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	Gene3D	G3DSA:3.90.870.10		1	217	5.1E-95T21-10-2019
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	SUPERFAMILY	SSF55821		7	213	5.95E-86T	21-10-2019	IPR017945	DHBP synthase RibB-like alpha/beta domain superfamily
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	Pfam	PF00926	3,4-dihydroxy-2-butanone 4-phosphate synthase	17	208	1.7E-82	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231
accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	PANTHER	PTHR21327		1	217	8.2E-126	T21-10-2019""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={"B7UIV5": {"GO:0008686", "GO:0009231"}},
      ),
      dict(
          testcase_name="two sequences, one has labels",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	PANTHER	PTHR21327		1	217	8.2E-126	T21-10-2019
    accession="Q5SMK6"	e9a286a263b71156fcf0cfebc12caec6	360	CDD	cd00143	PP2Cc	64	325	6.91138E-87	T	21-10-2019	IPR001932	PPM-type phosphatase domain	GO:0003824""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED
>accession="Q5SMK6"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={
              "B7UIV5": set(),
              "Q5SMK6": {"GO:0003824"},
          },
      ),
      dict(
          testcase_name="two sequences, both have labels",
          interproscan_output="""accession="B7UIV5"	74c763abf8567dfb6f4f83a4e0a31454	217	TIGRFAM	TIGR00506	ribB: 3,4-dihydroxy-2-butanone-4-phosphate synthase	13	209	1.5E-86	T	21-10-2019	IPR000422	3,4-dihydroxy-2-butanone 4-phosphate synthase, RibB	GO:0008686|GO:0009231
    accession="Q5SMK6"	e9a286a263b71156fcf0cfebc12caec6	360	CDD	cd00143	PP2Cc	64	325	6.91138E-87	T	21-10-2019	IPR001932	PPM-type phosphatase domain	GO:0003824""",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED
>accession="Q5SMK6"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={
              "B7UIV5": {"GO:0008686", "GO:0009231"},
              "Q5SMK6": {"GO:0003824"},
          },
      ),
      dict(
          testcase_name="sequence in ground truth that is missing in interproscan output",
          interproscan_output="",
          input_ground_truth_test_fasta=""">accession="B7UIV5"\tlabels="GO:101010"
NOT_USED""",
          expected_predicted_labels_per_seq={
              "B7UIV5": set(),
          },
      ),
  )
  def test_load_interproscan_output(self, interproscan_output,
                                    input_ground_truth_test_fasta,
                                    expected_predicted_labels_per_seq):
    # Set up inputs.
    input_file = _write_to_file(interproscan_output)

    input_test_fasta_filename = _write_to_file(input_ground_truth_test_fasta)

    input_ground_truth_test = baseline_utils.load_ground_truth(
        input_test_fasta_filename)

    # Compute actual results.
    actual_interproscan_output = baseline_utils.load_interproscan_output(
        test_data_ground_truth=input_ground_truth_test,
        interproscan_output_filename=input_file)

    # Assertions.
    expected_df_length = len(
        set(
            list(expected_predicted_labels_per_seq.keys()) +
            input_ground_truth_test.sequence_name.values))
    self.assertLen(actual_interproscan_output, expected_df_length)

    for row in actual_interproscan_output.itertuples():
      self.assertIn(row.sequence_name, expected_predicted_labels_per_seq)
      self.assertSetEqual(row.predicted_label,
                          expected_predicted_labels_per_seq[row.sequence_name])


if __name__ == "__main__":
  absltest.main()
