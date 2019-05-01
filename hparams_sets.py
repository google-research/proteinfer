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

r"""Hyperparameter sets.

These are defined as functions to allow for inheritance.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def _starting_hparams():
  """Set of shared starting parameters used in sets below."""
  hparams = tf.contrib.training.HParams()
  hparams.add_hparam('batch_style', 'bucket')
  hparams.add_hparam('gradient_clip', 1.0)
  hparams.add_hparam('learning_rate', 0.0005)
  hparams.add_hparam('lr_decay_rate', .997)
  hparams.add_hparam('lr_decay_steps', 1000)
  hparams.add_hparam('lr_warmup_steps', 3000)
  hparams.add_hparam('model_type', 'cnn')
  hparams.add_hparam('resnet_bottleneck_factor', 0.5)
  hparams.add_hparam('decision_threshold', 0.5)
  hparams.add_hparam('denominator_power', 1.0)  # Standard mean-pooling.
  return hparams


def tuned_for_ec():
  """Hyperparameters tuned for EC classification."""
  # TODO(theosanderson): update these to true SOTA values
  hparams = tf.contrib.training.HParams()
  hparams.add_hparam('batch_style', 'bucket')
  hparams.add_hparam('batch_size', 34)
  hparams.add_hparam('dilation_rate', 5)
  hparams.add_hparam('filters', 411)
  hparams.add_hparam('first_dilated_layer', 1)  # This is 0-indexed
  hparams.add_hparam('kernel_size', 7)
  hparams.add_hparam('num_layers', 5)
  hparams.add_hparam('pooling', 'mean')
  hparams.add_hparam('resnet_bottleneck_factor', 0.88152)
  hparams.add_hparam('lr_decay_rate', 0.9977)
  hparams.add_hparam('learning_rate', 0.00028748)
  hparams.add_hparam('decision_threshold', 0.3746)
  hparams.add_hparam('denominator_power', 0.88)

  hparams.add_hparam('train_steps', 650000)
  return hparams


def small_test_model():
  """A small test model that will run on a CPU quickly."""
  hparams = _starting_hparams()
  hparams.add_hparam('batch_size', 8)
  hparams.add_hparam('dilation_rate', 1)
  hparams.add_hparam('first_dilated_layer', 1)  # This is 0-indexed
  hparams.add_hparam('filters', 10)
  hparams.add_hparam('kernel_size', 3)
  hparams.add_hparam('num_layers', 1)
  hparams.add_hparam('train_steps', 100)
  return hparams
