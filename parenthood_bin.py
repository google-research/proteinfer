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
r"""Writes a dictionary of all transitive labels to apply to an example.

Includes EC, and GO labels (Enzyme Commission, Gene Ontology).

The format is gzipped json from key (GO term, EC number) to
list of all labels that should be applied to this term. For canonical labels,
the values in this list contain the key itself.
For non-canonical labels (e.g. obsoleted labels), the value of this map for a
key may not include the key itself, thus describing a normalized form for
labels.

Example usage:

mkdir data
cd data
wget http://current.geneontology.org/ontology/go.obo
wget ftp://ftp.expasy.org/databases/enzyme/enzclass.txt
wget ftp://ftp.expasy.org/databases/enzyme/enzyme.dat
cd ..
python parenthood_bin.py
cp /tmp/parenthood.json.gz data/

"""

import gzip
import io
import json
from absl import app
from absl import flags
from absl import logging

import parenthood_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('output_file', '/tmp/parenthood.json.gz',
                    'The file to which the parenthood file will be written')


def _get_ec_transitive():
  """Loads dictionary of label to implied labels for EC numbers."""
  logging.info('Getting EC parenthood dict.')

  with open(parenthood_lib.EC_LEAF_NODE_METADATA_PATH) as f:
    leaf_node_contents = f.read()
  with open(parenthood_lib.EC_NON_LEAF_NODE_METADATA_PATH) as f:
    non_leaf_node_contents = f.read()

  ec = parenthood_lib.parse_full_ec_file_to_transitive_parenthood(
      leaf_node_contents, non_leaf_node_contents)
  return ec


def _get_go_transitive():
  """Loads dictionary of label to implied labels for GO terms."""
  logging.info('Getting GO parenthood dict.')
  with open(parenthood_lib.GO_METADATA_PATH) as f:
    whole_go_contents = f.read()
  go_nontransitive = parenthood_lib.parse_full_go_file(whole_go_contents)

  go_transitive = parenthood_lib.transitive_go_parenthood(go_nontransitive)
  return go_transitive


def get_output_dict():
  """Get output dictionary of label to set of transitive applicable labels.

  Returns:
     Dictionary from label to all labels (transitively) that should be used
     for an example with that label.

  Raises:
    ValueError if go and ec terms contain any shared keys.
  """
  ec = _get_ec_transitive()
  go = _get_go_transitive()
  
  overlapping_keys = (
      set(ec.keys()).intersection(go.keys()))
  if overlapping_keys:
    raise ValueError('There was an overlap in keys between EC/GO. '
                     'Overlapping keys: {}'.format(overlapping_keys))

  to_write = dict()
  to_write.update(ec)
  to_write.update(go)
  return to_write


def write_output_dict(output_dict, output_path):
  """Writes `output_dict` as json to a gzipped file."""
  to_write_json = json.dumps(
      {k: sorted(list(v)) for k, v in output_dict.items()},
      sort_keys=True,
  )

  logging.info('gzipping dictionary.')
  gzip_contents = io.BytesIO()
  with gzip.GzipFile(fileobj=gzip_contents, mode='w') as f:
    f.write(to_write_json.encode('utf-8'))

  logging.info('Writing to file.')
  with open(output_path, 'wb') as output_file:
    output_file.write(gzip_contents.getvalue())


def main(_):
  output_dict = get_output_dict()
  write_output_dict(output_dict, FLAGS.output_file)

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
