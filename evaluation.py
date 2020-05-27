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

"""Evaluation utilities for uniprot predictions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Text, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy as pd_DataFrameGroupBy
import inference
import parenthood_lib
import sklearn.metrics
import tqdm

FALSE_NEGATIVES_KEY = 'false_negatives'
FALSE_POSITIVES_KEY = 'false_positives'
TRUE_POSITIVES_KEY = 'true_positives'

PrecisionRecallF1 = Tuple[float, float, float]


def normalize_confidences(
    predictions, label_vocab,
    applicable_label_dict):
  """Set confidences of parent labels to the max of their children.

  Args:
    predictions: [num_sequences, num_labels] ndarray.
    label_vocab: list of vocab strings in an order that corresponds to
      `predictions`.
    applicable_label_dict: Mapping from labels to their parents (including
      indirect parents).

  Returns:
    A numpy array [num_sequences, num_labels] with confidences where:
    if label_vocab[k] in applicable_label_dict[label_vocab[j]],
    then arr[i, j] >= arr[i, k] for all i.
  """
  vocab_indices = {v: i for i, v in enumerate(label_vocab)}
  children = parenthood_lib.reverse_map(applicable_label_dict,
                                        set(vocab_indices.keys()))

  # Only vectorize this along the sequences dimension as the number of children
  # varies between labels.
  label_confidences = []
  for label in label_vocab:
    child_indices = np.array(
        [vocab_indices[child] for child in children[label]])
    if child_indices.size > 1:
      confidences = np.max(predictions[:, child_indices], axis=1)
      label_confidences.append(confidences)
    else:
      label_confidences.append(predictions[:, vocab_indices[label]])

  return np.stack(label_confidences, axis=1)


def get_ground_truth_multihots(label_sets,
                               label_vocab):
  """Get a multihot matrix from label sets and a vocab."""
  vocab_indices = {v: i for i, v in enumerate(label_vocab)}
  ground_truths = []
  for s in label_sets:
    indices = np.array([vocab_indices[v] for v in s], dtype=np.int32)
    multihots = np.zeros([len(label_vocab)])
    multihots[indices] = 1
    ground_truths.append(multihots)
  return np.vstack(ground_truths)


def get_pr_f1_df_from_arrays(
    ground_truths,
    normalized_predictions,
    prediction_precision_limit = None,
):
  """Convenience method for making a PR curve dataframe.

  Args:
    ground_truths: multihot array of shape (num_examples, num_labels).
    normalized_predictions: array of shape (num_samples, num_labels).
    prediction_precision_limit: Used to truncate the predictions to a fixed
      level of precision. Predictions are truncated to
      `prediction_precision_limit` number of decimal places. This argument is
      useful to increase the speed of computation, and also to decrease the size
      of the dataframe returned. If None, no truncation is performed.

  Returns:
    pd.DataFrame with columns precision (float); recall (float);
    threshold (float); f1 (float).
  """
  if prediction_precision_limit:
    normalized_predictions = np.around(normalized_predictions,
                                       prediction_precision_limit)

  precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
      ground_truths.flatten(), normalized_predictions.flatten())
  # Throw away last precision and recall as they are always 0 and 1
  # respectively, and have no associated threshold
  precisions = precisions[:-1]
  recalls = recalls[:-1]
  f1s = 2 * (precisions * recalls) / (precisions + recalls)

  return pd.DataFrame(
      data={
          'precision': precisions,
          'recall': recalls,
          'threshold': thresholds,
          'f1': f1s
      })


def get_pr_f1_df(
    prediction_df,
    label_vocab,
    label_normalizer,
    eval_vocab = None,

    prediction_precision_limit = 18,
):
  """Make a dataframe with each possible threshold and it's corresponding values.

  Args:
    prediction_df: A dataframe with columns `predictions` and `true_label`.
    label_vocab: A list of labels.
    label_normalizer: A mapping from labels to their children.
    eval_vocab: An optional subset of `label_vocab` on which to restrict
      analysis.
    prediction_precision_limit: Used to truncate the predictions to a fixed
      level of precision. Predictions are truncated to
      `prediction_precision_limit` number of decimal places. This argument is
      useful to increase the speed of computation, and also to decrease
      the size of the dataframe returned. If None, no truncation is performed.

  Returns:
    A dataframe with 4 columns; precision, recall, f1, and threshold. At each
    threshold precision, recall, and f1 are calculated relative to the
    normalized confidences and true labels given in `prediction_df`.
  """
  if not eval_vocab:
    eval_vocab = set(label_vocab)
  label_vocab = np.array(label_vocab)

  prediction_array = np.vstack(prediction_df.predictions)
  normalized_predictions = normalize_confidences(prediction_array, label_vocab,
                                                 label_normalizer)
  true_label_sets = prediction_df.true_label.apply(
      eval_vocab.intersection).values
  eval_indices = np.array(
      [i for i, v in enumerate(label_vocab) if v in eval_vocab])
  ground_truths = get_ground_truth_multihots(true_label_sets,
                                             label_vocab[eval_indices])
  return get_pr_f1_df_from_arrays(ground_truths,
                                  normalized_predictions[:, eval_indices],
                                  prediction_precision_limit)


def true_false_positive_negative_df(df):
  """Computes df of all example/label pairs, and whether they were correct.

  Args:
    df: pd.Dataframe that has columns: true_label. Contains a set of true
      labels. predicted_label. Contains a set of true labels. sequence_name.
      string. Accession.

  Returns:
    pd.DataFrame that has columns:
      sequence_name. string. Name of sequence (accession).
      class. string. Class name. Either predicted or true.
      predicted. np.bool. Whether the class was predicted for the sequence.
      true. np.bool. Whether the class label is true for the sequence.
      true_positive. Whether the prediction is a true positive.
      false_positive. Whether the prediction is a false positive.
      false_negative. Whether the prediction is a false negative.
  """
  dict_prep_for_df = {
      'sequence_name': [],
      'class': [],
      'predicted': [],
      'true': []
  }

  for _, row in tqdm.tqdm(df.iterrows(), position=0, total=len(df)):
    all_classes = row.predicted_label.union(row.true_label)
    for cls in all_classes:
      dict_prep_for_df['sequence_name'].append(row.sequence_name)
      dict_prep_for_df['class'].append(cls)
      dict_prep_for_df['predicted'].append(cls in row.predicted_label)
      dict_prep_for_df['true'].append(cls in row.true_label)

  working_df = pd.DataFrame(dict_prep_for_df)

  working_df.predicted = working_df.predicted.astype(np.bool)
  working_df.true = working_df.true.astype(np.bool)

  working_df['true_positive'] = working_df.predicted & working_df.true
  working_df['false_positive'] = working_df.predicted & ~working_df.true
  working_df['false_negative'] = ~working_df.predicted & working_df.true

  return working_df


def multilabel_precision_per_example_label_pair(
    df):
  """Computes precision score of predictions in dataframe.

  Each (example, prediction) pair counts as "one" toward the precision. This is
  different than counting each class equally, or counting each example evenly.

  Args:
    df: E.g. output of true_false_positive_negative_df that has columns
      true_positive. bool. false_positive. bool.

  Returns:
    precision. Does not consider any thresholds.
  """
  true_positive = df.true_positive.sum().astype(float)
  false_positive = df.false_positive.sum().astype(float)
  return true_positive / (true_positive + false_positive)


def multilabel_recall_per_example_label_pair(
    df):
  """Computes f1 score of predictions in dataframe.

  Each (example, prediction) pair counts as "one" toward the recall. This is
  different than counting each class equally, or counting each example evenly.

  Args:
    df: E.g. output of true_false_positive_negative_df that has columns
      true_positive. bool. false_negative. bool.

  Returns:
    recall. Does not consider any thresholds.
  """
  true_positive = df.true_positive.sum().astype(float)
  false_negative = df.false_negative.sum().astype(float)
  return true_positive / (true_positive + false_negative)


def multilabel_f1_per_example_label_pair(
    df):
  """Computes f1 score of predictions in dataframe.

  Each (example, prediction) pair counts as "one" toward the f1 score. This is
  different than counting each class equally, or counting each example evenly.

  Args:
    df: has columns: true_positive. bool. false_positive. bool. false_negative.
      bool.

  Returns:
    f1 score. Harmonic mean of precision and recall. Does not consider any
    thresholds.
  """
  precision = multilabel_precision_per_example_label_pair(df)
  recall = multilabel_recall_per_example_label_pair(df)
  return 2. * (precision * recall) / (precision + recall)


def normalize_predictions(
    to_normalize,
    normalize_map):
  normalized_non_flattened = [normalize_map[e] for e in to_normalize]
  return frozenset(itertools.chain(*normalized_non_flattened))  # Flatten list.


def precision_recall_f1(
    df,
    label_normalizing_dict):
  """Returns precision, recall, and f1 for a dataframe.

  Args:
    df: pd.DataFrame with columns sequence_name, true__label, and
      predicted_label.
    label_normalizing_dict: dictionary of label to implied labels. Used to
      normalize labels into canonical form (e.g. an obsolete label is normalized
      into its replacement). See parenthood_lib.get_applicable_label_dict.

  Returns:
    PrecisionRecallF1.
  """
  blast_prepped = pd.DataFrame()
  blast_prepped['sequence_name'] = df.sequence_name
  blast_prepped['predicted_label'] = df.predicted_label.apply(
      lambda x: normalize_predictions(x, label_normalizing_dict))
  blast_prepped['true_label'] = df.true_label

  denormalized = true_false_positive_negative_df(blast_prepped)
  precision = multilabel_precision_per_example_label_pair(denormalized)
  recall = multilabel_recall_per_example_label_pair(denormalized)
  f1 = multilabel_f1_per_example_label_pair(denormalized)

  return precision, recall, f1


def filter_predictions_to_above_threshold(
    predictions, decision_threshold,
    label_vocab):
  """Computes predictions above `decision_threshold` for each example.

  Args:
    predictions: np.array, (2-d) of float. Outer dimension is example, inner
      dimension is class label. I.e. predictions[2, 3] is the probability that
      for example 2, the third class is true.
    decision_threshold: float. Classes with predictions above this threshold
      will be included in the output.
    label_vocab: np.array (1-d) of string. List of classes that corresponds to
      prediction.

  Returns:
    List of FrozenSet. Outer dimension is the example number within this batch,
    inner is a set of labels predicted that have confidence over
    decision_threshold.
  """
  preds_per_sequence = []
  for row in predictions:
    preds_per_sequence.append(
        frozenset(label_vocab[np.array(row) > decision_threshold]))

  return preds_per_sequence


def get_predictions_above_threshold(
    input_df,
    inferrer,
    decision_threshold,
    label_vocab = None):
  """Return df of predictions above a threshold for each sequence.

  Args:
    input_df: pd.DataFrame with columns sequence_name, sequence.
    inferrer: inferrer from a savedmodel model (see
      protein_task.MultiDiscreteLabelProteinTask) with activation_type
      'serving_default'.
    decision_threshold: float. Classes with predictions above this threshold
      will be included in the output.
    label_vocab: A numpy array with the string labels in vocab order. If None,
      will try to fetch this tensor from `inferrer`.

  Returns:
    pd.DataFrame with columns sequence_name, sequence, and predicted_label,
    where the values in column predicted_label are frozenset of labels
    that had confidences above decision_threshold.
  """
  if label_vocab is None:
    label_vocab = inferrer.get_variable('label_vocab:0')
  working_df = input_df.copy()

  df_with_predictions = inference.predictions_for_df(working_df, inferrer)

  preds_per_sequence = filter_predictions_to_above_threshold(
      predictions=df_with_predictions['predictions'].values,
      decision_threshold=decision_threshold,
      label_vocab=label_vocab)

  df_with_predictions.drop(columns=['predictions'], inplace=True)
  df_with_predictions['predicted_label'] = preds_per_sequence
  return df_with_predictions


def _family_and_clan_to_just_clan(
    family_and_clan):
  """Converts family_and_clan to just a clan if there is one.

  Args:
    family_and_clan: a set of either just a family, or a family and its
      associated clan.

  Returns:
    If family_and_clan is only a family, return family_and_clan. If
    family_and_clan has a clan, only return the clan.

  Raises:
    ValueError if len(family_and_clan != 1 or 2.
    Also raises if len(family-and_clan) == 2 and there's no clan in it.
  """
  if len(family_and_clan) == 1:
    return frozenset(family_and_clan)
  if len(family_and_clan) == 2:
    for f_or_c in family_and_clan:
      if f_or_c.startswith('Pfam:CL'):
        return frozenset([f_or_c])
    raise ValueError('family_and_clan was length 2, but did not have a clan in '
                     'it. family_and_clan was {}'.format(family_and_clan))

  raise ValueError('Expected either one or two values for family_and_clan. '
                   'Was {}'.format(family_and_clan))


def pfam_label_normalizer_to_lifted_clan(
    label_normalizer):
  """Converts label_normalizer (that may contan EC etc.) to lifted clans."""
  working_dict = copy.deepcopy(label_normalizer)
  working_dict = {k: v for k, v in working_dict.items() if k.startswith('Pfam')}
  return {k: _family_and_clan_to_just_clan(v) for k, v in working_dict.items()}


def convert_pfam_ground_truth_to_lifted_clans(
    ground_truth,
    label_normalizer):
  """Converts ground truth to only have labels that are lifted clans.

  The label normalizer may be already a lifted clan normalizer or it may be
  a non-lifted clan normalizer.

  Args:
    ground_truth: pd.DataFrame with columns sequence_name (str),
      true_label(Set[str]).
    label_normalizer: label normalizer. This will be converted to a lifted clan
      normalizer for use internally.

  Returns:
    pd.DataFrame with columns sequence_name (str), true_label(FrozenSet[str]).
  """
  lifted_label_normalizer = pfam_label_normalizer_to_lifted_clan(
      label_normalizer)
  working_df = ground_truth.copy()
  working_df['true_label'] = working_df.true_label.apply(
      lambda l: normalize_predictions(l, lifted_label_normalizer))
  return working_df


def _ec_label_at_level(label, level):
  """Return EC label up to and including level, or nan if it's a hyphen."""
  # nan is a useful value for pd.DataFrame that's used to indicate missing
  # data.
  label = label.replace('EC:', '')
  split = label.split('.')
  if split[level - 1] == '-':
    return np.nan
  return '.'.join(split[:level])


def ec_agreement_for_level(df,
                           level):
  """Returns agreement and disagreement between predictions and truth.

  Computes agreement, disagreement, and no-calls between true labels and
  predicted labels at level 1, 2, 3, or 4 in the EC hierarchy. If the ground
  truth label has a dash at this level, this is not included in our analysis
  of agreement/disagreement, as there's nothing to agree or disagree about.

  See the test for a more exhaustive listing of cases.

  Args:
    df: pd.DataFrame with columns 'true_label' (str) and 'predicted_label'
      (str). It's assumed that the input proteins are single-function enzymes,
      and that true and predicted label are the most specific predictions
      available (e.g. EC:1.2.3.4 instead of EC:1.2.3.-).
    level: int between 1 and 4.

  Returns:
    tuple of ints: [agreement, disagreement, no call made].

  Raises:
    ValueError if level is not between 1 and 4 inclusive.
  """
  if not (level >= 1 and level <= 4):
    raise ValueError(
        'Expected level to be between 1 and 4. Was {}'.format(level))
  get_ec_at_level = lambda x: _ec_label_at_level(x, level)
  true_labels = df.true_label.apply(get_ec_at_level)
  predicted_labels = df.predicted_label.apply(get_ec_at_level)

  has_ground_truth_label = true_labels.notna()
  pred_made = has_ground_truth_label & (predicted_labels.notna())

  agree = has_ground_truth_label & pred_made & (true_labels == predicted_labels)
  disagree = has_ground_truth_label & pred_made & (
      true_labels != predicted_labels)

  agree_numerator = sum(agree)
  disagree_numerator = sum(disagree)
  no_pred_made_numerator = sum(~pred_made & has_ground_truth_label)

  if (agree_numerator + disagree_numerator + no_pred_made_numerator !=
      sum(has_ground_truth_label)):
    # Internal correctness check, so raise an AssertionError, not a ValueError.
    error_msg = (f'Expected the sum agree_numerator + disagree_numerator + '
                 f'no_pred_made_numerator == sum(has_ground_truth_label): were '
                 f'{agree_numerator}, {disagree_numerator}, '
                 f'{no_pred_made_numerator} {sum(has_ground_truth_label)}')

    raise AssertionError(error_msg)
  return agree_numerator, disagree_numerator, no_pred_made_numerator
