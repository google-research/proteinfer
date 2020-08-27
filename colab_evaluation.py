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
"""Version of evaluation utilities intended for use in lower memory environments, such as colab."""

import numpy as np
import pandas as pd
import sklearn
import tqdm

import inference
import utils
import evaluation


def batch_inferences(iterator, batch_size):
    """Yield batches of seq_ids and predictions matrices from an iterator."""
    counter = 0
    predictions = []
    seq_ids = []
    while True:
        try:
            inference = next(iterator)
        except StopIteration:
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


def make_tidy_df_from_seq_names_and_prediction_array(
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
    """Take a directory of sharded inferences and output a tidy and normalized dataframe."""
    batches = batched_inferences_from_dir(shard_dir_path)
    dfs = []
    for seq_names, confidences in batches:
        normed_confidences = evaluation.normalize_confidences(
            confidences, vocab, label_normalizer)
        dfs.append(
            make_tidy_df_from_seq_names_and_prediction_array(
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
    """Perform an inner join of predictions and ground truth, then set all empty values to False."""
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


def filter_pr_curve(precisions, recalls, thresholds, resolution=1e-3):
    """Filters out imperceptible shifts in a PR curve."""
    last_precision = None
    last_recall = None
    new_precisions = []
    new_recalls = []
    new_thresholds = []
    for i in range(len(precisions)):
        if last_precision is None or abs(precisions[i] -
                                         last_precision) >= resolution or abs(
                                             recalls[i] -
                                             last_recall) >= resolution:
            new_precisions.append(precisions[i])
            last_precision = precisions[i]
            new_recalls.append(recalls[i])
            last_recall = recalls[i]
            new_thresholds.append(thresholds[i])
    return np.array(new_precisions), np.array(new_recalls), np.array(
        new_thresholds)


def assign_tp_fp_fn(predictions_df, ground_truth_df, threshold=0.5):
    """Assign each row of a predictions_df as either a TP,FP or FN."""
    combined = merge_predictions_and_ground_truth(predictions_df,
                                                  ground_truth_df)

    combined['tp'] = (combined['gt'] == True) & (combined['value'] > threshold)
    combined['fp'] = (combined['gt']
                      == False) & (combined['value'] > threshold)
    combined['fn'] = (combined['gt'] == True) & (combined['value'] < threshold)
    return combined
