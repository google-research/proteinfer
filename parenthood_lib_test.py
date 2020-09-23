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

"""Tests for module model_performance_analysis.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import textwrap

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import parenthood_lib
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class ParenthoodLibTest(parameterized.TestCase):

  def setUp(self):
    self._data_path = path.join(FLAGS.test_srcdir,
                                './testdata')
    super().setUp()

  @parameterized.named_parameters(
      dict(
          testcase_name='no relationships',
          input_text="""
              [Term]
              id: GO:0000001""",
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000001',
              term_name=None,
              description=None,
              related_labels=set()),
      ),
      dict(
          testcase_name='only is-a relationship',
          input_text="""
              [Term]
              id: GO:0000001
              is_a: GO:0048308 ! organelle inheritance""",
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000001',
              term_name=None,
              description=None,
              related_labels={('is_a', 'GO:0048308')}),
      ),
      dict(
          testcase_name='intersection_of relationship',
          input_text="""
              [Term]
              id: GO:0000018
              name: regulation of DNA recombination
              intersection_of: regulates GO:0006310 ! DNA recombination
              """,
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000018',
              term_name='regulation of DNA recombination',
              description=None,
              related_labels=set()),
      ),
      dict(
          testcase_name='replaced_by relationship',
          input_text="""
              [Term]
              id: GO:0000108
              name: obsolete repairosome
              comment: This term was made obsolete because 'repairosome' has fallen out of use in the literature, and the large complex described in the definition has not been confirmed to exist. The term has also confused annotators.
              synonym: "repairosome" EXACT []
              is_obsolete: true
              replaced_by: GO:0000109""",
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000108',
              term_name='obsolete repairosome',
              description=None,
              related_labels={('replaced_by', 'GO:0000109')}),
      ),
      dict(
          testcase_name='two is-a relationships',
          input_text="""
              [Term]
              id: GO:0000001
              is_a: GO:0048308 ! organelle inheritance
              is_a: GO:0048311 ! mitochondrion distribution""",
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000001',
              term_name=None,
              description=None,
              related_labels={('is_a', 'GO:0048308'),
                              ('is_a', 'GO:0048311')}),
      ),
      dict(
          testcase_name='no parent relationships, lots of others',
          input_text="""
              [Term]
              id: GO:0000001
              name: mitochondrion inheritance
              namespace: biological_process
              def: "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton." [GOC:mcc, PMID:10873824, PMID:11389764]
              synonym: "mitochondrial inheritance" EXACT []""",
          expected=parenthood_lib.GoTerm(
              term_id='GO:0000001',
              term_name='mitochondrion inheritance',
              description='The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton.',
              related_labels=set()),
      ),
  )
  def test_go_term_from_string(self, input_text, expected):
    input_text = textwrap.dedent(input_text)
    actual = parenthood_lib.GoTerm.from_string(input_text)
    self.assertEqual(actual, expected)

  def test_go_term_parsing_integration_test(self):
    gene_ontology_reference_full_file_path = path.join(self._data_path,
                                                       'go.obo')
    with tf.io.gfile.GFile(gene_ontology_reference_full_file_path, 'r') as f:
      full_go_contents = f.read()
    actual = parenthood_lib.parse_full_go_file(full_go_contents)

    expected_number_go_terms = 49933
    self.assertLen(actual, expected_number_go_terms)

    # Assert there are no duplicate keys in the labels.
    self.assertLen(actual, len(set(x.term_id for x in actual)))

    is_obsolete = (
        lambda term: any(l[0] == 'replaced_by' for l in term.related_labels))
    actual_non_obsolete_go_terms = [t for t in actual if not is_obsolete(t)]
    self.assertLess(len(actual_non_obsolete_go_terms), len(actual))

  def test_transitive_go_parenthood(self):
    child1 = parenthood_lib.GoTerm(
        term_id='child1',
        term_name=None,
        description=None,
        related_labels={('is_a', 'parent1'), ('is_a', 'parent2')})
    parent1 = parenthood_lib.GoTerm(
        term_id='parent1',
        term_name=None,
        description=None,
        related_labels={('is_a', 'grandparent')})
    parent2 = parenthood_lib.GoTerm(
        term_id='parent2',
        term_name=None,
        description=None,
        related_labels={('is_a', 'grandparent')})
    grandparent = parenthood_lib.GoTerm(
        term_id='grandparent',
        term_name=None,
        description=None,
        related_labels=set())

    child2 = parenthood_lib.GoTerm(
        term_id='child2',
        term_name=None,
        description=None,
        related_labels=set())
    input_parenthood_data = [child1, child2, parent1, parent2, grandparent]

    expected = {
        'child1': {'child1', 'parent1', 'parent2', 'grandparent'},
        'child2': {'child2'},
        'parent1': {'parent1', 'grandparent'},
        'parent2': {'parent2', 'grandparent'},
        'grandparent': {'grandparent'}
    }

    actual = parenthood_lib.transitive_go_parenthood(input_parenthood_data)

    self.assertDictEqual(actual, expected)

  def test_transitive_go_parenthood_with_cycle(self):
    loop1 = parenthood_lib.GoTerm(
        term_id='loop1',
        term_name=None,
        description=None,
        related_labels={('is_a', 'loop2')})

    loop2 = parenthood_lib.GoTerm(
        term_id='loop2',
        term_name=None,
        description=None,
        related_labels={('is_a', 'loop1')})

    input_parenthood_data = [loop1, loop2]

    expected = {
        'loop1': {'loop1', 'loop2'},
        'loop2': {'loop2', 'loop1'},
    }

    actual = parenthood_lib.transitive_go_parenthood(input_parenthood_data)

    self.assertDictEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='no hyphens, all single digits',
          label='1.2.3.4',
          expected={'1.2.3.4', '1.2.3.-', '1.2.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='one hyphen, all single digits',
          label='1.2.3.-',
          expected={'1.2.3.-', '1.2.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='two hyphens, all single digits',
          label='1.2.-.-',
          expected={'1.2.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='three hyphens, all single digits',
          label='1.-.-.-',
          expected={'1.-.-.-'},
      ),
      dict(
          testcase_name='no hyphens, multi digit',
          label='1.22.3.4',
          expected={'1.22.3.4', '1.22.3.-', '1.22.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='one hyphen, multi digit',
          label='1.22.3.-',
          expected={'1.22.3.-', '1.22.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='two hyphens, multi digit',
          label='1.22.-.-',
          expected={'1.22.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='no hyphens, contains n, single digit',
          label='1.2.3.n4',
          expected={'1.2.3.n4', '1.2.3.-', '1.2.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='one hyphen, contains n, multi digit',
          label='1.22.n3.-',
          expected={'1.22.n3.-', '1.22.-.-', '1.-.-.-'},
      ),
      dict(
          testcase_name='two hyphens, contains n, multi digit',
          label='1.n22.-.-',
          expected={'1.n22.-.-', '1.-.-.-'},
      ),
  )
  def test_all_ec_parents_for_term(self, label, expected):
    actual = parenthood_lib._all_ec_parents_for_label(label)
    self.assertEqual(actual, expected)

  def test_parse_full_ec_file_to_transitive_parenthood_integration_test(self):
    enzyme_commission_reference_dat_full_file_path = path.join(
        self._data_path, 'enzyme.dat')
    enzyme_commission_reference_class_full_file_path = path.join(
        self._data_path, 'enzclass.txt')

    # Load input data.
    with tf.io.gfile.GFile(enzyme_commission_reference_dat_full_file_path) as f:
      whole_ec_dat_contents = f.read()

    with tf.io.gfile.GFile(
        enzyme_commission_reference_class_full_file_path) as f:
      whole_ec_class_contents = f.read()

    # Compute actual output.
    actual = parenthood_lib.parse_full_ec_file_to_transitive_parenthood(
        whole_ec_dat_contents, whole_ec_class_contents)

    expected_to_contain = {  # Chosen somewhat arbitrarily.
        'EC:1.1.1.1',
        'EC:1.2.3.4',
        'EC:7.1.1.1',
        'EC:2.7.11.n2',
        'EC:1.-.-.-',

        # This label has no direct children in enzyme.dat, so we make sure it's
        # added in by the enzclass.txt file.
        'EC:3.12.-.-',

        # This entry was previously missing due to a bug, and so it's been added
        # here as a regression test.
        'EC:1.10.98.-',
    }
    self.assertContainsSubset(expected_to_contain, actual.keys())

    expected_length = 8163
    self.assertLen(actual, expected_length)

    expected_particular_parents = {
        'EC:1.2.3.4',
        'EC:1.2.3.-',
        'EC:1.2.-.-',
        'EC:1.-.-.-',
    }
    self.assertEqual(actual['EC:1.2.3.4'], expected_particular_parents)

  def test_ec_label_to_description(self):
    enzyme_commission_reference_dat_full_file_path = path.join(
        self._data_path, 'enzyme.dat')
    enzyme_commission_reference_class_full_file_path = path.join(
        self._data_path, 'enzclass.txt')

    # Load input data.
    with tf.io.gfile.GFile(enzyme_commission_reference_dat_full_file_path) as f:
      whole_ec_dat_contents = f.read()

    with tf.io.gfile.GFile(
        enzyme_commission_reference_class_full_file_path) as f:
      whole_ec_class_contents = f.read()

    actual = parenthood_lib.ec_label_to_description(whole_ec_dat_contents,
                                                    whole_ec_class_contents)
    self.assertIn('EC:1.-.-.-', actual)
    self.assertEqual(actual['EC:1.-.-.-'], 'Oxidoreductases.')

    self.assertIn('EC:1.2.3.4', actual)
    self.assertEqual(actual['EC:1.2.3.4'], 'Oxalate oxidase.')

  @parameterized.named_parameters(
      dict(
          testcase_name='one alt_id',
          input_go_term=parenthood_lib.GoTerm(
              term_id='term',
              term_name=None,
              description=None,
              related_labels={('alt_id', 'alt')}),
          expected=[
              parenthood_lib.GoTerm(
                  term_id='alt',
                  term_name=None,
                  description=None,
                  related_labels={
                      (parenthood_lib._IS_NON_CANONICAL_ALT_ID_LABEL_OF, 'term')
                  }),
              parenthood_lib.GoTerm(
                  term_id='term',
                  term_name=None,
                  description=None,
                  related_labels=set()),
          ]),
      dict(
          testcase_name='one replaced_by',
          input_go_term=parenthood_lib.GoTerm(
              term_id='go1',
              term_name='term name',
              description='Term description.',
              related_labels={('replaced_by', 'go2')}),
          expected=[
              # Should be a no-op.
              parenthood_lib.GoTerm(
                  term_id='go1',
                  term_name='term name',
                  description='Term description.',
                  related_labels={('replaced_by', 'go2')}),
          ]),
      dict(
          testcase_name='two alt_id',
          input_go_term=parenthood_lib.GoTerm(
              term_id='go1',
              term_name='Term name',
              description='Term description.',
              related_labels={('alt_id', 'go2'), ('alt_id', 'go3')}),
          expected=[
              parenthood_lib.GoTerm(
                  term_id='go2',
                  term_name='Term name',
                  description='Term description.',
                  related_labels={
                      (parenthood_lib._IS_NON_CANONICAL_ALT_ID_LABEL_OF, 'go1')
                  }),
              parenthood_lib.GoTerm(
                  term_id='go3',
                  term_name='Term name',
                  description='Term description.',
                  related_labels={
                      (parenthood_lib._IS_NON_CANONICAL_ALT_ID_LABEL_OF, 'go1')
                  }),
              parenthood_lib.GoTerm(
                  term_id='go1',
                  term_name='Term name',
                  description='Term description.',
                  related_labels=set()),
          ]),
      dict(
          testcase_name='one replaced_by with other non-identity relation',
          input_go_term=parenthood_lib.GoTerm(
              term_id='child',
              term_name=None,
              description=None,
              related_labels={('replaced_by', 'replacement'),
                              ('is_a', 'parent')}),
          expected=[
              # Should be a no-op.
              parenthood_lib.GoTerm(
                  term_id='child',
                  term_name=None,
                  description=None,
                  related_labels={('is_a', 'parent'),
                                  ('replaced_by', 'replacement')}),
          ]),
      dict(
          testcase_name='one alt_id with other non-identity relation',
          input_go_term=parenthood_lib.GoTerm(
              term_id='child',
              term_name=None,
              description=None,
              related_labels={('alt_id', 'alt'), ('is_a', 'parent')}),
          expected=[
              parenthood_lib.GoTerm(
                  term_id='alt',
                  term_name=None,
                  description=None,
                  related_labels={
                      (parenthood_lib._IS_NON_CANONICAL_ALT_ID_LABEL_OF,
                       'child')
                  }),
              parenthood_lib.GoTerm(
                  term_id='child',
                  term_name=None,
                  description=None,
                  related_labels={('is_a', 'parent')}),
          ]),
  )
  def test_yield_terms_for_alt_ids(self, input_go_term, expected):
    actual = list(parenthood_lib._yield_terms_for_alt_ids(input_go_term))

    self.assertCountEqual(actual, expected)

  def test_reverse_map_filters_items(self):
    test_parents = {'b': ['a', 'b'], 'a': ['a'], 'c': ['c'], 'd': ['a', 'b']}
    test_vocab = ['a', 'b', 'c']
    rev = parenthood_lib.reverse_map(test_parents, label_vocab=test_vocab)
    rev_map = {'a': {'a', 'b'}, 'b': {'b'}, 'c': {'c'}}
    self.assertEqual(rev, rev_map)

  def test_is_implied_by_something_else_positive_case(self):
    input_label = 'CL0192'
    input_reversed_normalizer = {
        'CL0192': {'PF00001', 'PF00002'},
        'PF00002': {'PF00002'},
    }
    input_other_labels_for_protein = {'CL0192', 'PF00002'}
    actual = parenthood_lib.is_implied_by_something_else(
        input_label, input_reversed_normalizer, input_other_labels_for_protein)
    expected = True
    self.assertEqual(actual, expected)

  def test_is_implied_by_something_else_negative_case(self):
    input_label = 'CL0192'
    input_reversed_normalizer = {
        'CL0192': {'PF00001', 'PF00002'},
    }
    input_other_labels_for_protein = {'CL0192'}
    actual = parenthood_lib.is_implied_by_something_else(
        input_label, input_reversed_normalizer, input_other_labels_for_protein)
    expected = False
    self.assertEqual(actual, expected)

  def test_filter_labels_to_most_specific(self):
    input_df = pd.DataFrame({
        'predicted_label': [
            frozenset(['CL0192', 'PF00002']),
            frozenset(['CL0192'])
        ]
    })
    input_normalizer = {
        'PF00002': frozenset(['PF00002', 'CL0192']),
        'CL0192': frozenset(['CL0192'])
    }

    actual = parenthood_lib.filter_labels_to_most_specific(
        input_df, input_normalizer)
    actual_predicted_labels = actual.predicted_label.values.tolist()

    expected = [frozenset(['PF00002']), frozenset(['CL0192'])]

    self.assertListEqual(actual_predicted_labels, expected)


if __name__ == '__main__':
  absltest.main()
