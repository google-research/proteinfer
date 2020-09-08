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
"""Version of evaluation utilities intended for use in lower memory environments, such as colab.

This version of the evaluation code leverages the fact that the vast majority of example-label predictions 
are essentially zero, and so only contribute to false negatives. It therefore represents the data in "tidy
format" with one row per example-label pair and excludes example-label pairs below a defined threshold.
"""

import numpy as np
import pandas as pd
import sklearn
import tqdm

import inference
import utils
import evaluation

def read_blast_table(filename):
  blast_out = pd.read_table(filename,names=['up_id','target',  'pc_identity', 'pc_positives', 'alignment_length', 'mismatches', 'gap_opens', 'q. start', 'q. end', 's. start', 'evalue', 'bit_score'])

  def extract_accession(long_string):
      return long_string.replace('accession="','').replace('"','')
  
  blast_out['up_id']=blast_out['up_id'].map(extract_accession)
  blast_out['target']=blast_out['target'].map(extract_accession)
  blast_out = blast_out[['up_id','target','pc_identity','alignment_length','bit_score']]
  return blast_out

def stats_by_group(df):
    """Calculate statistics from a groupby'ed dataframe with TPs,FPs and FNs."""

    EPSILON = 1e-10
    result = df[['tp', 'fp', 'fn']].sum().reset_index().assign(
        precision=lambda x: (x['tp'] + EPSILON) / (x['tp'] + x['fp'] + EPSILON),
        recall=lambda x: (x['tp'] + EPSILON) / (x['tp'] + x['fn'] + EPSILON)
    ).assign(
        f1=
        lambda x: 2 * x['precision'] * x['recall'] / (x['precision'] + x['recall'] + EPSILON),
        count=lambda x: x['tp'] + x['fn'])
    result['proportion'] = result['count'] / np.sum(result['count'])
    result['proportion_text'] = (
        result['proportion'] * 100).round(2).astype(str) + "%"
    return result

def get_stats(df):
    """Calculate statistics from a dataframe with TPs,FPs and FNs."""
    df['dummy_group'] = 'all'
    data = stats_by_group(df.groupby('dummy_group')).drop(
        columns=['dummy_group', 'proportion', 'proportion_text'])
    return data

def apply_threshold_and_return_stats(predictions_df,
                                     ground_truth_df,
                                     threshold=0.5,
                                     grouping=None):
    """Given predictions, ground truth and a threshold get statistics."""
    calls = assign_tp_fp_fn(predictions_df, ground_truth_df, threshold)
    if grouping:
        calls['group'] = calls['label'].map(grouping)
        return stats_by_group(
            calls.groupby("group")).assign(threshold=threshold)
    else:
        return get_stats(calls).assign(threshold=threshold) 

def batch_inferences(iterator, batch_size):
    """Yield batches of seq_ids and predictions matrices from an iterator."""
    counter = 0
    predictions = []
    seq_ids = []
    while True:
        try:
            inference = next(iterator)
        except StopIteration:
            if len(seq_ids) > 0:
                yield seq_ids, np.vstack(predictions)
            return
        seq_ids.append(inference[0])
        predictions.append(inference[1])
        counter += 1
        if counter == batch_size:
            yield seq_ids, np.vstack(predictions)
            predictions = []
            seq_ids = []
            counter = 0


def batched_inferences_from_files(shard_names, batch_size=100):
    """Iterate through TFRecord files of inferences and yield batches."""
    for file_name in tqdm.tqdm(shard_names, position=0):
        inference_iterator = inference.parse_shard(file_name)
        batched_iterator = batch_inferences(inference_iterator, batch_size)
        while True:
            try:
                yield next(batched_iterator)
            except StopIteration:
                break


def batched_inferences_from_dir(shard_dir_path, batch_size=100):
    """Iterate through directory of inference TFRecord files and yield batches."""
    files_to_process = utils.absolute_paths_of_files_in_dir(shard_dir_path)
    return batched_inferences_from_files(files_to_process, batch_size)


def _make_tidy_df_from_seq_names_and_prediction_array(
        sequence_names,
        predictions_array,
        vocab,
        min_decision_threshold=1e-20):
    """Given a list of sequences and a matrix of prediction values, yield a tidy dataframe of predictions."""
    up_ids = []
    labels = []
    values = []

    for i in range(len(sequence_names)):
        up_id = sequence_names[i]
        preds = predictions_array[i, :]

        for vocab_index in np.argwhere(preds > min_decision_threshold):
            vocab_index = vocab_index[0]
            up_ids.append(up_id)
            labels.append(vocab[vocab_index])
            values.append(preds[vocab_index])
    return pd.DataFrame({"up_id": up_ids, "label": labels, "value": values})


def get_normalized_inference_results(shard_dir_path,
                                     vocab,
                                     label_normalizer,
                                     min_decision_threshold=1e-20):
    """Take a directory of sharded inferences and output a tidy and normalized dataframe.

    Inferences are in the format defined in inference.py
    
    Args:
        shard_dir_path: directory of TFrecord inference shards
        vocab: a list of vocabulary items
        label_normalizer: a dictionary mapping vocabulary items to their parents
        min_decision_threshold: a threshold reflecting the minimum we will ever be 
            able to use to call a positive in subsequent analysis. Higher
            values use less RAM at the expense of lower maximum sensitivity.
        
    Returns:
        A pandas dataframe with one row per example-label (provided value > min_decision_threshold) and the 
        associated value from the neural network.
    """
    batches = batched_inferences_from_dir(shard_dir_path)
    dfs = []
    for seq_names, confidences in batches:
        normed_confidences = evaluation.normalize_confidences(
            confidences, vocab, label_normalizer)
        dfs.append(
            _make_tidy_df_from_seq_names_and_prediction_array(
                seq_names,
                normed_confidences,
                vocab,
                min_decision_threshold=min_decision_threshold))
    return pd.concat(dfs)


def make_tidy_df_from_ground_truth(ground_truth):
    """Create a tidy dataframe from ground truth data."""
    up_ids = []
    labels = []

    for i in tqdm.tqdm(ground_truth.index, position=0):
        up_id = ground_truth['sequence_name'][i]
        for vocab_entry in ground_truth['true_label'][i]:
            up_ids.append(up_id)
            labels.append(vocab_entry)
    return pd.DataFrame({"up_id": up_ids, "label": labels, "gt": True})


def merge_predictions_and_ground_truth(predictions_df, ground_truth_df):
    """Perform an outer join of predictions and ground truth, then set all empty values to False."""
    combined = predictions_df.merge(ground_truth_df,
                                    how="outer",
                                    suffixes=("_pred", "_gt"),
                                    left_on=["label", "up_id"],
                                    right_on=["label", "up_id"])
    combined = combined.fillna(False)
    return combined


def get_pr_curve_df(predictions_df,
                    ground_truth_df,
                    grouping=None,
                    filtered=True):
    """Given predictions and ground truth in tidy format, yield a precision recall curve.
    
    Args:
        predictions_df: predictions in tidy format
        ground_truth_df: ground truth in tidy format
        grouping: optional dictionary mapping sequence names to categories
        filtered: whether to remove almost redundant points on PR curve
    """
    combined = merge_predictions_and_ground_truth(predictions_df,
                                                  ground_truth_df)
    if grouping == None:
        to_process = {'all': combined}.items()
    else:
        combined['group'] = combined['label'].map(grouping)
        to_process = combined.groupby('group')

    del combined
    output_dfs = []
    for group_name, group in tqdm.tqdm(to_process, position=0):
        precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(
            group['gt'], group['value'])
        precisions = precisions[:-1]
        recalls = recalls[:-1]
        if filtered:
            precisions, recalls, thresholds = filter_pr_curve(
                precisions, recalls, thresholds)
        output_dfs.append(
            pd.DataFrame({
                'group':
                group_name,
                'precision':
                precisions,
                'recall':
                recalls,
                'threshold':
                thresholds,
                'f1':
                2 * precisions * recalls / (precisions + recalls)
            }))
    return pd.concat(output_dfs)


def filter_pr_curve(precisions, recalls, thresholds, resolution=1e-4):
    """Filters out imperceptible shifts in a PR curve."""
    last_precision = None
    last_recall = None
    new_precisions = []
    new_recalls = []
    new_thresholds = []
    for i in range(len(precisions)):
        if last_precision is None  or abs(
                                             recalls[i] -
                                             last_recall) >= resolution:
            new_precisions.append(precisions[i])
            last_precision = precisions[i]
            new_recalls.append(recalls[i])
            last_recall = recalls[i]
            new_thresholds.append(thresholds[i])
    return np.array(new_precisions), np.array(new_recalls), np.array(
        new_thresholds)


def assign_tp_fp_fn(predictions_df, ground_truth_df, threshold):
    """Return a new predictions dataframe where each row is assigned as either a TP, FP or FN."""
    combined = merge_predictions_and_ground_truth(predictions_df,
                                                  ground_truth_df)

    combined['tp'] = (combined['gt'] == True) & (combined['value'] > threshold)
    combined['fp'] = (combined['gt']
                      == False) & (combined['value'] > threshold)
    combined['fn'] = (combined['gt'] == True) & (combined['value'] < threshold)
    return combined
