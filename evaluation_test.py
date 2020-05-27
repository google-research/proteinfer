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
"""Tests for evaluation.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import evaluation
from six.moves import range


class _InferrerFixture(object):
  activation_type = "serving_default"

  def __init__(self, vocab_size):
    self._vocab_size = vocab_size

  def get_activations(self, l):
    # Return the first l rows of the identity matrix.
    return np.eye(self._vocab_size)[:len(l)]

  def get_variable(self, _):
    return np.array(["CLASS_{}".format(i) for i in range(self._vocab_size)])


class TestEvaluation(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="one_label",
          confidences=np.array([[1.0], [1.0]]),
          vocab=["a"],
          normalized=np.array([[1.0], [1.0]]),
          parents={"a": ["a"]}),
      dict(
          testcase_name="missing_implied_label",
          confidences=np.array([[1.0], [1.0]]),
          vocab=["a"],
          normalized=np.array([[1.0], [1.0]]),
          parents={"a": ["a", "b"]}),
      dict(
          testcase_name="one_example",
          confidences=np.array([[0.6, 0.4]]),
          vocab=["a", "b"],
          normalized=np.array([[0.6, 0.6]]),
          parents={
              "a": ["a", "b"],
              "b": ["b"]
          }))
  def test_normalize_confidences_edge(self, confidences, vocab, normalized,
                                      parents):
    test_normalized = evaluation.normalize_confidences(confidences, vocab,
                                                       parents)
    np.testing.assert_array_equal(test_normalized, normalized)

  def test_normalize_confidences(self):
    test_confidences = np.array([[0.2, 0.5, 0.1, 0.2], [0.1, 0.5, 0.2, 0.1],
                                 [0.1, 0.1, 0.6, 0.2]])
    test_vocab = ["a", "b", "c", "d"]
    test_parents = {"b": ["a", "b", "c"], "d": ["d"], "a": ["a"], "c": ["c"]}

    normalized = evaluation.normalize_confidences(test_confidences, test_vocab,
                                                  test_parents)

    for child, parents in test_parents.items():
      children = normalized[:, test_vocab.index(child)]
      for parent in parents:
        non_violations = (children <= normalized[:, test_vocab.index(parent)])
        self.assertTrue(
            np.all(non_violations),
            msg=(
                "Parent: {} violates parenthood invariant at indices:\n "
                # pylint: disable=g-explicit-bool-comparison, singleton-comparison
                "{}".format(parent, np.where(non_violations == False))))

  def test_get_ground_truth_multihots(self):
    label_sets = [{"foo"}, {"foo", "bar"}, {"bar", "baz"}, set()]
    label_vocab = sorted(["foo", "bar", "baz"])

    expected = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]])
    multihots = evaluation.get_ground_truth_multihots(label_sets, label_vocab)
    np.testing.assert_array_equal(expected, multihots)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_subset",
          # After normalization predictions are:
          # [0.4, 0.3, 0.1], [0.4, 0.4, 0.3]
          # Threshold of .3 gives  4 correct out of 5
          expected_precision=np.array([4. / 5, 1.0]),
          # Threshold of .4 gives 3 found out of 4.
          expected_recall=np.array([1.0, 3. / 4])),
      dict(
          testcase_name="subset",
          eval_subset={"a", "c"},
          # Without label "b" all correct
          expected_precision=np.array([1.0, 1.0]),
          # Threshold of 0.4 gives 2 out of 3 positive labels.
          expected_recall=np.array([1.0, 2.0 / 3])))
  def test_get_pr_f1_df(self,
                        expected_precision,
                        expected_recall,
                        eval_subset=None):
    df = pd.DataFrame(
        dict(
            true_label=[{"a"}, {"a", "b", "c"}],
            predictions=[[0.4, 0.3, 0.1], [0.1, 0.4, 0.3]]))
    test_vocab = ["a", "b", "c"]
    test_parents = {"b": ["a", "b"], "a": ["a"], "c": ["c"]}

    # After normalization predictions are:
    # [0.4, 0.3, 0.1], [0.4, 0.4, 0.3]
    expected_thresholds = np.array([0.3, 0.4])
    expected_f1 = ((2 * expected_recall * expected_precision) /
                   (expected_recall + expected_precision))
    pr_f1_df = evaluation.get_pr_f1_df(
        df, test_vocab, test_parents, eval_vocab=eval_subset)
    np.testing.assert_array_equal(pr_f1_df.precision.values, expected_precision)
    np.testing.assert_array_equal(pr_f1_df.recall.values, expected_recall)
    np.testing.assert_array_equal(pr_f1_df.threshold.values,
                                  expected_thresholds)
    np.testing.assert_array_equal(pr_f1_df.f1.values, expected_f1)

  def test_get_pr_f1_df_precision_truncation(self):
    df = pd.DataFrame(
        dict(
            true_label=[{"a", "b", "c"}],
            predictions=[np.array([0.0001, 0.0002, 0.0003])]))
    test_vocab = ["a", "b", "c"]
    test_parents = {"a": ["a"], "b": ["b"], "c": ["c"]}

    # We expect that when truncating precision, the precision recall df is
    # shorter.
    pr_f1_df_unlimited_precision = evaluation.get_pr_f1_df(
        df, test_vocab, test_parents, prediction_precision_limit=None)
    pr_f1_df_limited_precision = evaluation.get_pr_f1_df(
        df, test_vocab, test_parents, prediction_precision_limit=1)

    self.assertLess(
        len(pr_f1_df_limited_precision), len(pr_f1_df_unlimited_precision),
        "{}\nis not shorter than\n{}".format(
            pr_f1_df_limited_precision,
            pr_f1_df_unlimited_precision,
        ))

  @parameterized.named_parameters(
      dict(
          testcase_name="one class, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_1"}],
                  true_label=[{"CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="one class, imperfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_1"}],
                  true_label=[{"CLASS_0"}])),
          expected=0.,
      ),
      dict(
          testcase_name="two classes, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0", "CLASS_1"}],
                  true_label=[{"CLASS_0", "CLASS_1"}])),
          expected=1.),
      dict(
          testcase_name="two classes, too many predictions, all recalled",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0", "CLASS_1"}],
                  true_label=[{"CLASS_0"}])),
          expected=1.,
      ),
      dict(
          testcase_name="two classes, not enough predictions",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0"}],
                  true_label=[{"CLASS_0", "CLASS_1"}])),
          expected=.5,
      ),
      dict(
          testcase_name="two examples, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0"}, {"CLASS_1"}],
                  true_label=[{"CLASS_0"}, {"CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="two examples, imperfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0"}, {"CLASS_0"}],
                  true_label=[{"CLASS_0"}, {"CLASS_1"}])),
          expected=.5,
      ),
      dict(
          testcase_name="two examples, one with two labels, one not recalled",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0"}, {"CLASS_1"}],
                  true_label=[{"CLASS_0", "CLASS_1"}, {"CLASS_1"}])),
          expected=2. / 3,
      ),
  )
  def test_multilabel_recall_per_example_label_pair(self, df, expected):
    working_df = evaluation.true_false_positive_negative_df(df)
    actual = evaluation.multilabel_recall_per_example_label_pair(working_df)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="one class, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_1"}],
                  true_label=[{"CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="one class, imperfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_1"}],
                  true_label=[{"CLASS_0"}])),
          expected=0.,
      ),
      dict(
          testcase_name="two classes, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0", "CLASS_1"}],
                  true_label=[{"CLASS_0", "CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="two classes, too many predictions",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0", "CLASS_1"}],
                  true_label=[{"CLASS_0"}])),
          expected=.5,
      ),
      dict(
          testcase_name="two classes, not enough predictions",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1"],
                  predicted_label=[{"CLASS_0"}],
                  true_label=[{"CLASS_0", "CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="two examples, perfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0"}, {"CLASS_1"}],
                  true_label=[{"CLASS_0"}, {"CLASS_1"}])),
          expected=1.,
      ),
      dict(
          testcase_name="two examples, imperfect",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0"}, {"CLASS_0"}],
                  true_label=[{"CLASS_0"}, {"CLASS_1"}])),
          expected=.5,
      ),
      dict(
          testcase_name="two examples, one with two predicted labels, one wrong",
          df=pd.DataFrame(
              dict(
                  sequence_name=["seq1", "seq2"],
                  predicted_label=[{"CLASS_0", "CLASS_1"}, {"CLASS_1"}],
                  true_label=[{"CLASS_0"}, {"CLASS_1"}])),
          expected=2. / 3,
      ),
  )
  def test_multilabel_precision_per_example_label_pair(self, df, expected):
    working_df = evaluation.true_false_positive_negative_df(df)
    actual = evaluation.multilabel_precision_per_example_label_pair(working_df)
    self.assertEqual(actual, expected)

  def test_multilabel_f1_per_example_label_pair(self):
    df = pd.DataFrame(
        dict(
            sequence_name=["seq1", "seq2"],
            predicted_label=[{"CLASS_0", "CLASS_1"}, {"CLASS_1"}],
            true_label=[{"CLASS_0"}, {"CLASS_1"}]))
    df = evaluation.true_false_positive_negative_df(df)
    expected = .8  # Harmonic mean of 1. and 2./3
    actual = evaluation.multilabel_f1_per_example_label_pair(df)
    self.assertEqual(expected, actual)

  def test_precision_recall_f1_integration_test(self):
    df = pd.DataFrame(
        dict(
            sequence_name=["seq1", "seq2"],
            predicted_label=[{"CLASS_0", "CLASS_1"}, {"CLASS_1"}],
            true_label=[{"CLASS_1"}, {"CLASS_1"}]))

    normalizing_dict = {
        "CLASS_0": list(),  # Not included in output.
        "CLASS_1": ["CLASS_1"]
    }

    actual_precision, actual_recall, actual_f1 = evaluation.precision_recall_f1(
        df, normalizing_dict)

    expected_precision = 1.  # CLASS_0 is normalized out.
    expected_recall = 1.  # Both sequences have their true label recalled.
    expected_f1 = 1.

    self.assertEqual(actual_precision, expected_precision)
    self.assertEqual(actual_recall, expected_recall)
    self.assertEqual(actual_f1, expected_f1)

  @parameterized.named_parameters(
      dict(
          testcase_name="no sequences",
          input_predictions=np.array([]),
          decision_threshold=1.,
          label_vocab=np.array([]),
          expected=[],
      ),
      dict(
          testcase_name="two sequences, no predictions make it above threshold",
          input_predictions=np.array([[.5, .5], [.5, .5]]),
          decision_threshold=1.,
          label_vocab=np.array(["class1", "class2"]),
          expected=[frozenset(), frozenset()],
      ),
      dict(
          testcase_name="two sequences, one prediction makes it above threshold",
          input_predictions=np.array([[1., .5], [.5, .5]]),
          decision_threshold=.75,
          label_vocab=np.array(["class1", "class2"]),
          expected=[frozenset(["class1"]), frozenset()],
      ),
      dict(
          testcase_name="two sequences, one prediction from each makes it above threshold",
          input_predictions=np.array([[1., .5], [.5, 1.]]),
          decision_threshold=.75,
          label_vocab=np.array(["class1", "class2"]),
          expected=[frozenset(["class1"]),
                    frozenset(["class2"])],
      ),
      dict(
          testcase_name="two sequences, all classes are predicted",
          input_predictions=np.array([[1., 1.], [1., 1.]]),
          decision_threshold=.75,
          label_vocab=np.array(["class1", "class2"]),
          expected=[
              frozenset(["class1", "class2"]),
              frozenset(["class1", "class2"])
          ],
      ),
  )
  def test_filter_predictions_to_above_threshold(self, input_predictions,
                                                 decision_threshold,
                                                 label_vocab, expected):
    actual = evaluation.filter_predictions_to_above_threshold(
        predictions=input_predictions,
        decision_threshold=decision_threshold,
        label_vocab=label_vocab)

    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="medium threshold, all make the cut",
          input_seqs=["AAAA", "DDD", "EE", "W"],
          decision_threshold=.5,
          vocab_size=10,

          # Inferrer fixture outputs values of 1, so everything is above .5.
          expected_count_over_threshold=4,
      ),
      dict(
          testcase_name="high threshold, all make the cut",
          input_seqs=["AAAA", "DDD", "EE", "W"],
          decision_threshold=999,
          vocab_size=10,

          # Inferrer fixture outputs values of 1, so everything is below 999.
          expected_count_over_threshold=0,
      ),
  )
  def test_get_predictions_above_threshold(self, input_seqs, decision_threshold,
                                           vocab_size,
                                           expected_count_over_threshold):
    input_df = pd.DataFrame({
        "sequence_name": input_seqs,
        "sequence": input_seqs
    })
    inferrer_fixture = _InferrerFixture(vocab_size)

    actual = evaluation.get_predictions_above_threshold(
        input_df=input_df,
        inferrer=inferrer_fixture,
        decision_threshold=decision_threshold)

    # Assert columns are correct.
    expected_columns = ["sequence_name", "sequence", "predicted_label"]
    self.assertCountEqual(actual.columns, expected_columns)

    # Assert predicted_label values are all type frozenset.
    for predicted_label in actual.predicted_label.values:
      self.assertEqual(type(predicted_label), frozenset)

    # Assert number of called labels is correct.
    actual_count_over_threshold = len(
        [s for s in actual.predicted_label if len(s) > 0])  # pylint: disable=g-explicit-length-test
    self.assertEqual(actual_count_over_threshold, expected_count_over_threshold)

  @parameterized.named_parameters(
      dict(
          testcase_name="empty dict",
          input_normalizer={},
          expected={},
      ),
      dict(
          testcase_name="one entry",
          input_normalizer={"Pfam:PF00001": frozenset(["Pfam:PF00001"])},
          expected={"Pfam:PF00001": frozenset(["Pfam:PF00001"])},
      ),
      dict(
          testcase_name="one entry with clan",
          input_normalizer={
              "Pfam:PF00001": frozenset(["Pfam:PF00001", "Pfam:CL0192"])
          },
          expected={"Pfam:PF00001": frozenset(["Pfam:CL0192"])},
      ),
  )
  def test_pfam_label_normalizer_to_lifted_clan(self, input_normalizer,
                                                expected):
    actual = evaluation.pfam_label_normalizer_to_lifted_clan(input_normalizer)
    self.assertDictEqual(actual, expected)

    # Assert that pfam_label_normalizer_to_lifted_clan is idempotent by calling
    # the function again.
    second_actual = evaluation.pfam_label_normalizer_to_lifted_clan(
        input_normalizer)
    self.assertDictEqual(second_actual, actual)

  def test_pfam_label_normalizer_to_lifted_clan_raises_when_wrong_num(self):
    with self.assertRaisesRegex(ValueError, "one or two"):
      # This label implies 3 labels, which is not allowed.
      evaluation.pfam_label_normalizer_to_lifted_clan(
          {"Pfam:PF00001": frozenset(["Pfam:ONE", "Pfam:TWO", "Pfam:THREE"])})

  def test_pfam_label_normalizer_to_lifted_clan_raises_when_no_clan(self):
    with self.assertRaisesRegex(ValueError, "did not have a clan"):
      # This label implies 2 labels, none of which is a clan
      evaluation.pfam_label_normalizer_to_lifted_clan({
          "Pfam:PF00001": frozenset(["Pfam:NOT_A_CLAN_1", "PFam:NOT_A_CLAN_2"])
      })

  def test_convert_pfam_ground_truth_to_lifted_clans(self):
    input_df = pd.DataFrame({
        "sequence_name": ["SEQ1", "SEQ2"],
        "true_label": [
            frozenset(["Pfam:PF00001"]),
            frozenset(["Pfam:PF99999", "Pfam:PF88888"])
        ]
    })
    input_label_normalizer = {
        "Pfam:PF00001": frozenset(["Pfam:PF00001", "Pfam:CL0192"]),
        "Pfam:PF99999": frozenset(["Pfam:PF99999"]),
        "Pfam:PF88888": frozenset(["Pfam:PF88888"])
    }
    actual = evaluation.convert_pfam_ground_truth_to_lifted_clans(
        input_df, input_label_normalizer)

    # Expect PF00001 was converted to only CL0192, everything else stays the
    # same.
    expected = pd.DataFrame({
        "sequence_name": ["SEQ1", "SEQ2"],
        "true_label": [
            frozenset(["Pfam:CL0192"]),
            frozenset(["Pfam:PF99999", "Pfam:PF88888"])
        ]
    })

    np.testing.assert_array_equal(actual.sequence_name.values,
                                  expected.sequence_name.values)
    self.assertSetEqual(actual.true_label.values[0],
                        expected.true_label.values[0])
    self.assertSetEqual(actual.true_label.values[1],
                        expected.true_label.values[1])

  @parameterized.named_parameters(
      dict(
          testcase_name="agree at this level",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:1.1.1.1"],
              "true_label": ["EC:1.1.1.2"],
          }),
          input_level=1,
          expected_agree=1,
          expected_disagree=0,
          expected_no_call=0,
      ),
      dict(
          testcase_name="disagree at this level",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:1.1.1.1"],
              "true_label": ["EC:1.1.1.2"],
          }),
          input_level=4,
          expected_agree=0,
          expected_disagree=1,
          expected_no_call=0,
      ),
      dict(
          testcase_name="ground truth had no prediction at this level, but we did",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:1.1.1.1"],
              "true_label": ["EC:1.1.1.-"],
          }),
          input_level=4,
          expected_agree=0,
          expected_disagree=0,
          expected_no_call=0,
      ),
      dict(
          testcase_name="ground truth had no prediction at this level, and neither did we",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:1.1.1.-"],
              "true_label": ["EC:1.1.1.-"],
          }),
          input_level=4,
          expected_agree=0,
          expected_disagree=0,
          expected_no_call=0,
      ),
      dict(
          testcase_name="ground truth had no prediction at this level, we did make a prediction, but we made a mistake earlier",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:2.2.2.2"],
              "true_label": ["EC:1.1.1.-"],
          }),
          input_level=4,
          expected_agree=0,
          # We shouldn't be penalized for making a prediction at a level
          # where the ground truth also didn't make a prediction.
          expected_disagree=0,
          expected_no_call=0,
      ),
      dict(
          testcase_name="ground truth had prediction at this level, we did not make a prediction, and we made no mistake earlier",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:1.1.1.-"],
              "true_label": ["EC:1.1.1.1"],
          }),
          input_level=4,
          expected_agree=0,
          # We shouldn't be penalized for making no prediction, even if we
          # made a mistake higher up.
          expected_disagree=0,
          expected_no_call=1,
      ),
      dict(
          testcase_name="ground truth had prediction at this level, we did not make a prediction, but we made a mistake earlier",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:2.2.2.-"],
              "true_label": ["EC:1.1.1.1"],
          }),
          input_level=4,
          expected_agree=0,
          # We shouldn't be penalized for making no prediction, even if we
          # made a mistake higher up.
          expected_disagree=0,
          expected_no_call=1,
      ),
      dict(
          testcase_name="ground truth had no prediction at this level, and we didn't make a prediction at this level, but we made a mistake earlier",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:2.2.2.-"],
              "true_label": ["EC:1.1.1.-"],
          }),
          input_level=4,
          expected_agree=0,
          # We shouldn't be penalized for making a non-prediction at a level
          # where the ground truth also didn't make a prediction.
          expected_disagree=0,
          expected_no_call=0,
      ),
      dict(
          testcase_name="we made a mistake earlier on, but happened to agree on a lesser classification",
          input_df=pd.DataFrame({
              "predicted_label": ["EC:2.1.1.1"],
              "true_label": ["EC:1.1.1.1"],
          }),
          input_level=3,
          expected_agree=0,
          # Even though we agree that the 3rd level is a '1', because we made
          # a mistake in the first level, this should be counted as a
          # disagreement.
          expected_disagree=1,
          expected_no_call=0,
      ),
      dict(
          testcase_name="more than one prediction",
          input_df=pd.DataFrame({
              "predicted_label": [
                  "EC:1.1.1.1", "EC:1.1.1.1", "EC:1.1.1.1", "EC:1.1.1.-"
              ],
              "true_label": [
                  "EC:1.1.1.1", "EC:1.1.1.2", "EC:1.1.1.-", "EC:1.1.1.-"
              ],
          }),
          input_level=4,
          # First element is an agreement.
          expected_agree=1,
          # Second element is a disagreement.
          expected_disagree=1,
          # Last two elements are discarded because they're not called by the
          # ground truth.
          expected_no_call=0,
      ),
  )
  def test_ec_agreement_for_level(self, input_level, input_df, expected_agree,
                                  expected_disagree, expected_no_call):
    actual_agree, actual_disagree, actual_no_call = evaluation.ec_agreement_for_level(
        input_df, input_level)

    self.assertEqual(actual_agree, expected_agree,
                     "agreement numbers were not the same.")
    self.assertEqual(actual_disagree, expected_disagree,
                     "disagreement numbers were not the same.")
    self.assertEqual(actual_no_call, expected_no_call,
                     "no-call numbers were not the same.")


if __name__ == "__main__":
  absltest.main()
