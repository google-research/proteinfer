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

"""Construct model and evaluation metrics for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import protein_dataset
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.layers.python.layers import optimizers as optimizers_lib


_THRESHOLDS_FOR_RECALL_METRIC = [2, 3, 5, 10]
REPRESENTATION_KEY = 'representation'
LOGITS_KEY = 'logits'


def _f1_score(labels, predictions):
  """Computes F1 score, i.e. the harmonic mean of precision and recall."""
  precision = tf.metrics.precision(labels, predictions)
  recall = tf.metrics.recall(labels, predictions)

  return (2 * precision[0] * recall[0] / (precision[0] + recall[0] + 1e-5),
          tf.group(precision[1], recall[1]))


def _mean_examplewise_f1_score(labels, predictions):
  """Calculates mean example-wise F1 score (micro-F1).

  Args:
    labels: 2D tensor of one hots.
    predictions: 2D tensor of one hots.

  Returns:
    metric, update ops from tf.metrics.mean
  """
  labels = tf.cast(labels, tf.float32)
  predictions = tf.cast(predictions, tf.float32)

  assert len(labels.shape) == 2
  assert len(predictions.shape) == 2

  true_positives = labels * predictions
  false_positives = predictions * (1 - labels)
  false_negatives = (1 - predictions) * labels

  true_positives = tf.reduce_sum(true_positives, axis=1)
  false_positives = tf.reduce_sum(false_positives, axis=1)
  false_negatives = tf.reduce_sum(false_negatives, axis=1)

  precision = true_positives / (true_positives + false_positives + 1e-5)
  recall = true_positives / (true_positives + false_negatives)
  f1 = 2 * precision * recall / (precision + recall + 1e-5)
  # F1 score is not defined where there are no correct labels, ignore these:
  well_defined = tf.greater(true_positives + false_negatives, 0)

  # Remove any nans (these new 0s will be ignored by the weights anyway):
  f1 = tf.where(well_defined, f1, tf.zeros_like(f1))
  return tf.metrics.mean(f1, weights=well_defined)


def _custom_recall_at_k(labels_as_multi_hot, predictions, k):
  """Calculates recall_at_k metric with multi-hot labels.

  For each example which contains at least one label, a recall-at-k is
  calculated by assessing what proportion of these labels are in the top k
  predictions. This metric is the mean of these values.

  Args:
    labels_as_multi_hot: a tensor of [batch_size, num_output_classes] where
      elements are zero (absent) or one (present).
    predictions: a tensor of [batch_size, num_output_classes] where elemenents
      are floats indicating the probability of class membership.
    k: number of top predictions to consider (must be <= num_output_classes).

  Returns:
    mean: A scalar `Tensor` representing the current mean, the value of `total`
       divided by `count` (of finite values).
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose (scalar) value matches the mean_value.
  """
  labels_as_multi_hot = tf.cast(labels_as_multi_hot, tf.float32)

  num_output_classes = tf.shape(labels_as_multi_hot)[1]
  _, indices = tf.math.top_k(predictions, k=k)

  predictions_top_k_as_multi_hot = _indices_to_multihot(indices,
                                                        num_output_classes)

  true_positives_tensor = tf.math.logical_and(
      tf.cast(labels_as_multi_hot, tf.bool),
      tf.cast(predictions_top_k_as_multi_hot, tf.bool))

  false_negatives_tensor = tf.math.greater(labels_as_multi_hot,
                                           predictions_top_k_as_multi_hot)

  true_positives_per_example = tf.count_nonzero(true_positives_tensor, axis=1)
  false_negatives_per_example = tf.count_nonzero(false_negatives_tensor, axis=1)

  recall_per_example = true_positives_per_example / (
      true_positives_per_example + false_negatives_per_example)

  is_finite = tf.is_finite(recall_per_example)  # To filter out no label cases.
  recall_per_example_finite_only = tf.boolean_mask(recall_per_example,
                                                   is_finite)

  return tf.metrics.mean(recall_per_example_finite_only)


def _make_evaluation_metrics(labels, predictions, num_output_classes, hparams):
  """Construct various eval metrics.

  Args:
    labels: dict with ground truth data necessary for computing metrics.
    predictions: dict containing Tensors for predictions.
    num_output_classes: number of different labels.
    hparams: tf.contrib.HParams object.

  Returns:
    A dict where the values obey the tf.metrics API.
  """
  labels_op = labels[protein_dataset.LABEL_KEY]
  multi_hot_labels = _indices_to_multihot(labels_op, num_output_classes)
  predictions_as_floats = predictions[protein_dataset.LABEL_KEY]
  recall_threshold = hparams.decision_threshold
  predictions_as_bools = tf.greater(predictions_as_floats,
                                    tf.constant(recall_threshold))

  metrics = {
      'precision_at_threshold':
          tf.metrics.precision(
              labels=multi_hot_labels, predictions=predictions_as_bools),
      'recall_at_threshold':
          tf.metrics.recall(
              labels=multi_hot_labels, predictions=predictions_as_bools),
      'f1_at_threshold':
          _f1_score(labels=multi_hot_labels, predictions=predictions_as_bools),
      'mean_examplewise_f1_at_threshold':
          _mean_examplewise_f1_score(
              labels=multi_hot_labels, predictions=predictions_as_bools),
      'true_positives':
          tf.metrics.true_positives(
              labels=multi_hot_labels, predictions=predictions_as_bools),
      'false_positives':
          tf.metrics.false_positives(
              labels=multi_hot_labels, predictions=predictions_as_bools)
  }
  for k in _THRESHOLDS_FOR_RECALL_METRIC:
    metrics['recall@%d' % k] = _custom_recall_at_k(
        labels_as_multi_hot=multi_hot_labels,
        predictions=predictions_as_floats,
        k=k)

  return metrics


def _set_padding_to_sentinel(padded_representations, sequence_lengths,
                             sentinel):
  """Set padding on batch of padded representations to a sentinel value.

  Useful for preparing a batch of sequence representations for max or average
  pooling.

  Args:
    padded_representations: float32 tensor, shape (batch, longest_sequence, d),
      where d is some arbitrary embedding dimension. E.g. the output of
      tf.data.padded_batch.
    sequence_lengths: tensor, shape (batch,). Each entry corresponds to the
      original length of the sequence (before padding) of that sequence within
      the batch.
    sentinel: float32 tensor, shape: broadcastable to padded_representations.

  Returns:
    tensor of same shape as padded_representations, where all entries
      in the sequence dimension that came from padding (i.e. are beyond index
      sequence_length[i]) are set to sentinel.
  """
  sequence_dimension = 1
  embedding_dimension = 2

  with tf.variable_scope('set_padding_to_sentinel', reuse=False):
    longest_sequence_length = tf.shape(
        padded_representations)[sequence_dimension]
    embedding_size = tf.shape(padded_representations)[embedding_dimension]

    seq_mask = tf.sequence_mask(sequence_lengths, longest_sequence_length)
    seq_mask = tf.expand_dims(seq_mask, [embedding_dimension])
    is_not_padding = tf.tile(seq_mask, [1, 1, embedding_size])

    full_sentinel = tf.zeros_like(padded_representations)
    full_sentinel = full_sentinel + tf.convert_to_tensor(sentinel)

    per_location_representations = tf.where(
        is_not_padding, padded_representations, full_sentinel)

    return per_location_representations


def _make_per_sequence_features(per_location_representations, raw_features,
                                hparams):
  """Aggregate representations across the sequence dimension."""

  sequence_lengths = raw_features[protein_dataset.SEQUENCE_LENGTH_KEY]
  per_location_representations = _set_padding_to_sentinel(
      per_location_representations, sequence_lengths, tf.constant(0.))
  # We average the representations across the sequence length dimension:
  # tf.reduce_mean(..., axis=1) is problematic, since different batches
  # may be dynamically padded to different lengths. Instead, we normalize
  # each element of the batch individually, by the length of each element's
  # un-normalized sequence. We raise this to a tunable power to allow the
  # tuner to choose between mean and sum-pooling or an intermediate type.
  denominator = tf.cast(
      tf.expand_dims(
          raw_features[protein_dataset.SEQUENCE_LENGTH_KEY], axis=-1),
      tf.float32)**hparams.denominator_power

  pooled_representation = tf.reduce_sum(
      per_location_representations, axis=1) / denominator

  pooled_representation = tf.identity(
      pooled_representation, name='pooled_representation')

  return pooled_representation


def _convert_representation_to_prediction_ops(representation, raw_features,
                                              num_output_classes, hparams):
  """Map per-location features to problem-specific prediction ops.

  Args:
    representation: [batch_size, sequence_length, feature_dim] Tensor.
    raw_features: dictionary containing the raw input Tensors; this is the
      sequence, keyed by sequence_key.
    num_output_classes: number of different labels.
    hparams: tf.contrib.HParams object.

  Returns:
    predictions: dictionary containing Tensors that Estimator
      will return as predictions.
    predictions_for_loss: Tensor that make_loss() consumes.
  """
  per_sequence_features = _make_per_sequence_features(
      per_location_representations=representation,
      raw_features=raw_features,
      hparams=hparams)
  logits = tf.layers.dense(
      per_sequence_features, num_output_classes, name=LOGITS_KEY)

  predictions = {
      protein_dataset.LABEL_KEY:
          tf.identity(tf.sigmoid(logits), name='predictions')
  }

  predictions_for_loss = logits
  return predictions, predictions_for_loss


def _make_representation(features, hparams, mode):
  """Produces [batch_size, sequence_length, embedding_dim] features.

  Args:
    features: dict from str to Tensor, containing sequence and sequence length.
    hparams: tf.contrib.training.HParams()
    mode: tf.estimator.ModeKeys instance.

  Returns:
    Tensor of shape [batch_size, sequence_length, embedding_dim].
  """
  sequence_features = features[protein_dataset.SEQUENCE_KEY]
  sequence_lengths = features[protein_dataset.SEQUENCE_LENGTH_KEY]

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  sequence_features = _conv_layer(
      sequence_features=sequence_features,
      sequence_lengths=sequence_lengths,
      num_units=hparams.filters,
      dilation_rate=1,
      kernel_size=hparams.kernel_size,
  )

  for layer_index in range(hparams.num_layers):
    sequence_features = _residual_block(
        sequence_features=sequence_features,
        sequence_lengths=sequence_lengths,
        hparams=hparams,
        layer_index=layer_index,
        activation_fn=tf.nn.relu,
        is_training=is_training)

  return sequence_features


def _make_prediction_ops(features, hparams, mode, label_vocab):
  """Returns (predictions, predictions_for_loss, representation)."""
  representation = _make_representation(features, hparams, mode)

  representation = tf.identity(representation, name=REPRESENTATION_KEY)

  # Used to save constants in the graph, e.g. for SavedModel.
  _ = tf.constant(label_vocab, name='label_vocab')
  _ = tf.constant(hparams.decision_threshold, name='decision_threshold')

  num_output_classes = len(label_vocab)

  predictions, prediction_for_loss = _convert_representation_to_prediction_ops(
      representation=representation,
      raw_features=features,
      num_output_classes=num_output_classes,
      hparams=hparams)
  return predictions, prediction_for_loss


def _batch_norm(features, is_training):
  return tf.layers.batch_normalization(features, training=is_training)


def _conv_layer(sequence_features, sequence_lengths, num_units, dilation_rate,
                kernel_size):
  """Return a convolution of the input features that respects sequence len."""
  padding_zeroed = _set_padding_to_sentinel(sequence_features, sequence_lengths,
                                            tf.constant(0.))
  conved = tf.layers.conv1d(
      padding_zeroed,
      filters=num_units,
      kernel_size=[kernel_size],
      dilation_rate=dilation_rate,
      padding='same')

  # Re-zero padding, because shorter sequences will have their padding
  # affected by half the width of the convolution kernel size.
  re_zeroed = _set_padding_to_sentinel(conved, sequence_lengths,
                                       tf.constant(0.))
  return re_zeroed


def _residual_block(sequence_features, sequence_lengths, hparams, layer_index,
                    activation_fn, is_training):
  """Construct a single block for a residual network."""

  with tf.variable_scope('residual_block_{}'.format(layer_index), reuse=False):
    shifted_layer_index = layer_index - hparams.first_dilated_layer + 1
    dilation_rate = max(1, hparams.dilation_rate**shifted_layer_index)

    num_bottleneck_units = math.floor(
        hparams.resnet_bottleneck_factor * hparams.filters)

    features = _batch_norm(sequence_features, is_training)
    features = activation_fn(features)
    features = _conv_layer(
        sequence_features=features,
        sequence_lengths=sequence_lengths,
        num_units=num_bottleneck_units,
        dilation_rate=dilation_rate,
        kernel_size=hparams.kernel_size,
    )
    features = _batch_norm(features, is_training=is_training)
    features = activation_fn(features)

    # The second convolution is purely local linear transformation across
    # feature channels, as is done in
    # third_party/tensorflow_models/slim/nets/resnet_v2.bottleneck
    residual = _conv_layer(
        features,
        sequence_lengths,
        num_units=hparams.filters,
        dilation_rate=1,
        kernel_size=1)

    with_skip_connection = sequence_features + residual
    return with_skip_connection


def _indices_to_multihot(indices, vocab_size):
  """Converts [batch,n_labels] of indices to [batch,vocab_size] multihot.

  Indices can be padded with -1.

  Args:
    indices: dense tensor of indices [batch, arbitrary_n_labels], padded with -1
      if necessary.
    vocab_size: integer vocab_size.

  Returns:
    Multihot float32 tensor of dimension [batch, vocab_size].

  e.g. [[0,1],[2,-1]] (vocab_size:4) -> [1,1,0,0], [0,0,1,0]

  """

  if len(indices.shape) != 2:
    raise ValueError(
        'indices_to_multihot expects tensors of dimension 2, got shape %s' %
        indices.shape)

  sparse_indices = contrib_layers.dense_to_sparse(indices, eos_token=-1)

  multihot = tf.sparse.to_indicator(sparse_indices, vocab_size=vocab_size)

  multihot = tf.cast(multihot, tf.float32)
  return multihot


def _make_loss(predictions_for_loss, labels, num_output_classes):
  """Make scalar loss."""
  logits = predictions_for_loss
  labels_op = labels[protein_dataset.LABEL_KEY]

  # We need to get labels into a multi-hot format:
  labels_op = _indices_to_multihot(labels_op, vocab_size=num_output_classes)

  loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_op, logits=logits)
  loss = tf.reduce_mean(loss)
  return loss


def _make_train_op(loss, hparams):
  """Create train op."""

  def learning_rate_decay_fn(learning_rate, global_step):
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               hparams.lr_decay_steps,
                                               hparams.lr_decay_rate)
    learning_rate = learning_rate * tf.minimum(
        tf.cast(global_step / hparams.lr_warmup_steps, tf.float32),
        tf.constant(1.))
    return learning_rate

  return contrib_layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      clip_gradients=optimizers_lib.adaptive_clipping_fn(
          decay=hparams.gradient_clipping_decay,
          report_summary=True,
      ),
      learning_rate=hparams.learning_rate,
      learning_rate_decay_fn=learning_rate_decay_fn,
      optimizer='Adam')


def make_model_fn(label_vocab, hparams):
  """Returns a model function for estimator given prediction base class.

  Args:
    label_vocab: list of string.
    hparams: tf.contrib.HParams object.

  Returns:
    A function that returns a tf.estimator.EstimatorSpec
  """

  def _model_fn(features, labels, params, mode=None):
    """Returns tf.estimator.EstimatorSpec."""

    predictions, predictions_for_loss = _make_prediction_ops(
        features=features, hparams=params, mode=mode, label_vocab=label_vocab)

    evaluation_hooks = []
    num_output_classes = len(label_vocab)
    if mode == tf.estimator.ModeKeys.TRAIN:
      loss = _make_loss(
          predictions_for_loss=predictions_for_loss,
          labels=labels,
          num_output_classes=num_output_classes)
      train_op = _make_train_op(loss=loss, hparams=params)
      eval_ops = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
      loss = None
      train_op = None
      eval_ops = None
    else:  # Eval mode.
      loss = _make_loss(
          predictions_for_loss=predictions_for_loss,
          labels=labels,
          num_output_classes=num_output_classes)

      train_op = None
      eval_ops = _make_evaluation_metrics(
          labels=labels,
          predictions=predictions,
          num_output_classes=num_output_classes,
          hparams=hparams)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_ops,
        evaluation_hooks=evaluation_hooks,
    )

  return _model_fn
