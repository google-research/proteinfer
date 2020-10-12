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

import protein_dataset
import inference
import tqdm
import numpy as np
from absl import flags,app
import glob
import os
import utils


flags.DEFINE_string(
    'dataset', None,
    'Dataset')

flags.DEFINE_string('output_location', None,
                    'Output location')

flags.DEFINE_string('model_location', '',
                    'Model location')

      
FLAGS = flags.FLAGS
def generate_predictions(globbable_dataset, output_location, model_location, batch_size=12):

  
  sequence_iterator = protein_dataset.yield_examples(globbable_dataset)
  sequences = []
  labels = []
  ids = []
  for example in tqdm.tqdm(sequence_iterator):
    aas = set(example[protein_dataset.SEQUENCE_KEY])
    if aas.issubset(utils.AMINO_ACID_VOCABULARY):
      ids.append(example[protein_dataset.SEQUENCE_ID_KEY])
      sequences.append(example[protein_dataset.SEQUENCE_KEY])
      labels.append(example[protein_dataset.LABEL_KEY])

  # If we want to optimise for inference speed we should sort the dataset by
  # sequence length:
  seq_lengths = [len(x) for x in sequences]
  indices = np.argsort(-np.array(seq_lengths)).tolist()

  ids = [ids[indices[x]] for x in range(len(indices))]
  sequences = [sequences[indices[x]] for x in range(len(indices))]
  labels = [set(labels[indices[x]]) for x in range(len(indices))]

  model_location = glob.glob(f"{model_location}/*/")[0]

  inferrer = inference.Inferrer(model_location, use_tqdm= True,batch_size=batch_size)

  # Note that because sequences are sorted from longest to shortest this will be
  # much slower at first - do not despair (it should take a few minutes)

  dataset_writer = open(output_location,"wt")
  embeddings  = inferrer.get_activations(sequences)
  for i,embedding in enumerate(embeddings):
    embed_line = ",".join([str(x) for x in embedding])
    dataset_writer.write(f"{ids[i]}\t{embed_line}\n")



def main(_):
  generate_predictions(FLAGS.dataset, FLAGS.output_location, FLAGS.model_location)


if __name__ == '__main__':
  FLAGS.alsologtostderr = True  # Shows training output.

  app.run(main)
