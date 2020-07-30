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
# Lint as: python3
"""Utilities for parsing EC and GO labels, and finding their parents."""


import collections
import gzip
import itertools
import json
import os
import re
import typing
from typing import (Dict, FrozenSet, Iterable, List, Optional, Set, Text, Tuple,
                    Collection)

import pandas as pd
import tqdm


# From ftp://ftp.expasy.org/databases/enzyme/enzyme.dat
DATA_DIR = ''
EC_LEAF_NODE_METADATA_PATH = os.path.join(DATA_DIR, 'enzyme.dat')

# From ftp://ftp.expasy.org/databases/enzyme/enzclass.txt
EC_NON_LEAF_NODE_METADATA_PATH = os.path.join(DATA_DIR, 'enzclass.txt')

# From http://purl.obolibrary.org/obo/go.obo
GO_METADATA_PATH = os.path.join(DATA_DIR, 'go.obo')

# Labels that are implied by other labels.
# Json is a map from string key (label) to list of applicable/implied
# labels (string).
APPLICABLE_LABEL_JSON_PATH = os.path.join(DATA_DIR, 'parenthood.json.gz')


# GO:followed by seven digits, followed by
# either a space, or the end of the line.
GO_TERM_REGEX = re.compile(r'(GO:\d\d\d\d\d\d\d)( |$)')

# Allows numbers in all positions, or a hyphen in latter positions, or an 'n',
# indicating that the number is still undergoing consideration and review:
# https://www.ebi.ac.uk/ena/WebFeat/qualifiers/EC_number.html
EC_NUMBER_REGEX = r'(\d+).([\d\-n]+).([\d\-n]+).([\d\-n]+)'
_TOP_LEVEL_EC_CLASS_VALUE = '-.-.-.-'

# Format of lines at ftp://ftp.expasy.org/databases/enzyme/enzclass.txt
# that contain an EC number.
_NON_LEAF_NODE_LINE_REGEX = re.compile(r'^\d\.')


# The determination that these terms are either parenthood or not was made
# with the help of this document:
# https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_2.html
_IDENTITY_TYPE_GO_RELATIONS = {
    # Used for obsolete terms.
    'replaced_by',

    # Is basically a synonym.
    'alt_id',
}
_PARENTHOOD_TYPE_GO_RELATIONS = {
    # Clearly a transitive parenthood relation.
    'is_a',

}
_NON_PARENTHOOD_TYPE_GO_RELATIONS = {
    # -------- Tag types that are clearly not parenthood relations -----------
    'comment',
    'created_by',
    'creation_date',
    'def',
    'disjoint_from',
    'id',
    'is_obsolete',
    'name',
    'namespace',
    'property_value',

    # -------- Tag types that seem like they'd be parenthood relations -------
    # Gives a term which may be an appropriate substitute for an obsolete
    # term, but needs to be looked at carefully by a human expert before the
    # replacement is done.
    'consider',

    # Describes subsets of go terms that may be useful for different uses.
    'subset',

    # Contains references to databases other than GO terms.
    'synonym',

    # As of August 10 2019, the following are the types of relationships
    # listed in the canonical GO ontology OBO file:
    # {'ends_during',
    #  'happens_during',
    #  'has_part',
    #  'negatively_regulates',
    #  'occurs_in',
    #  'part_of',
    #  'positively_regulates',
    #  'regulates'}
    # None of these qualifies for propagation to children because of "is-a"
    # type semantics.
    'relationship',

    # Contains references to databases other than GO terms.
    'xref',

    # "intersection_of" This tag indicates that this term is equivalent to the
    # intersection of several other terms. Many times, one of the terms
    # is already in an is_a relationship, and the other intersection term has
    # a non-parenthood type relation, e.g. has-a. For this reason, we exclude
    # intersection_of relationships.
    'intersection_of',
}

_IS_NON_CANONICAL_ALT_ID_LABEL_OF = 'was_alt_id_of'

TermID = Text
RelationDescriptor = Text

# GO is an ontology of relations of "term" "related to" "other term".
# A GoAttribute describes the right-hand-side of this.
GoAttribute = Tuple[RelationDescriptor, TermID]


class GoTerm(
    typing.NamedTuple('GoTerm',
                      (('term_id', Text), ('term_name', Optional[Text]),
                       ('description', Optional[Text]),
                       ('related_labels', Set[GoAttribute])))):
  """A Gene Ontology term.

  A Gene Ontology term is a term that's related to other terms via an enumerated
  set of relation types. Not all relations are "is-a" relations.

  [1] Describes the format that go terms are presented in.

  [1] https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_2.html

  Attributes:
    term_id: id of term. E.g. GO:0000108
    term_name: value of "name:" tag in obo file.
    description: value of "def:" tag in obo file.
    related_labels: set of [relation, term_id] that also apply to this term.
      These labels are NOT transitive.
  """

  @classmethod
  def from_string(cls, s: Text) -> 'GoTerm':
    """Parses a Term block of OBO file, keeping identity and parent relations.

     https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_2.html

    Args:
      s: block of OBO file that starts with [Term]

    Returns:
      GoTerm
    """
    lines = s.split('\n')
    attributes = [
        _parse_go_attribute(l) for l in lines if _is_go_attribute_line(l)
    ]
    term_ids = [value for tag, value in attributes if tag == 'id']
    if len(term_ids) != 1:
      raise ValueError(('Number of term names for term was {} '
                        '(expected exactly one). Term was {}.').format(
                            len(term_ids), attributes))
    term_id = term_ids[0]

    related_labels = set()
    term_description = None
    term_name = None
    for tag, value in attributes:
      if tag == 'name':
        term_name = value

      if tag == 'def':
        term_description = re.findall('"(.*)" .*', value)[0]

      if tag in _PARENTHOOD_TYPE_GO_RELATIONS or tag in _IDENTITY_TYPE_GO_RELATIONS:
        related_labels.add((tag, _get_go_term_from_text(value)))
      elif tag in _NON_PARENTHOOD_TYPE_GO_RELATIONS:
        continue
      else:
        valid_relations = _PARENTHOOD_TYPE_GO_RELATIONS.union(
            _NON_PARENTHOOD_TYPE_GO_RELATIONS).union(
                _IDENTITY_TYPE_GO_RELATIONS)
        raise ValueError('Term type unknown: was {} and expected one of {}. '
                         'Full value was {}'.format(tag, valid_relations, s))

    return GoTerm(
        term_id=term_id,
        term_name=term_name,
        description=term_description,
        related_labels=related_labels)


def _is_go_attribute_line(s: Text) -> bool:
  return ': ' in s


def _parse_go_attribute(s: Text) -> GoAttribute:
  split = s.split(': ')
  return (split[0], ''.join(split[1:]))


def _get_go_term_from_text(s: Text) -> Text:
  matches = GO_TERM_REGEX.findall(s)
  if len(matches) != 1:
    raise ValueError(
        'Expected exactly one match for a GO term in string {}. Found matches {}'
        .format(s, matches))
  # First match, looking at the go term, not the (space or end-of-line).
  return matches[0][0]


def _yield_terms_for_alt_ids(term: GoTerm) -> Iterable[GoTerm]:
  """Yields GoTerms that point to the root term for all alt_ids in `term`.

  Alt ids do not have their own term in the ontology, so this function is used
  to create these terms and to canonicalize these alternative ids to their
  preferred ids.

  Args:
    term: GoTerm. May or may not have alt_ids in its parents.

  Yields:
    GoTerm for each alt_id of `term`, whose parent labels are only the
    label of term.
  """
  for relation, related_term_id in term.related_labels:
    if relation == 'alt_id':
      related_labels = {(_IS_NON_CANONICAL_ALT_ID_LABEL_OF, term.term_id)}
      non_canonical_term = GoTerm(related_term_id, term.term_name,
                                  term.description, related_labels)
      yield non_canonical_term

  yield GoTerm(
      term_id=term.term_id,
      term_name=term.term_name,
      description=term.description,
      related_labels={l for l in term.related_labels if l[0] != 'alt_id'})


def parse_full_go_file(file_contents: Optional[Text] = None) -> List[GoTerm]:
  """Parses contents of OBO file containing the GO ontology.

  Args:
    file_contents: string. File contents of go file.

  Returns:
    List of GoTerm.
  """

  if file_contents is None:
    with open(GO_METADATA_PATH) as f:
      file_contents = f.read()

  unparsed_terms = [
      x for x in file_contents.split('\n\n') if x.startswith('[Term]')
  ]
  parsed_terms = [GoTerm.from_string(t) for t in unparsed_terms]
  with_alt_ids_itr = itertools.chain(*(_yield_terms_for_alt_ids(x)
                                       for x in parsed_terms))

  return list(with_alt_ids_itr)


def go_label_to_description(
    go_file_contents: Optional[Text] = None) -> Dict[Text, Text]:
  return {
      t.term_id: f'{t.term_name}: {t.description}'
      for t in parse_full_go_file(go_file_contents)
  }


def _go_term_applicable_labels_should_include_themselves(term: GoTerm) -> bool:
  """Return whether this go term is canonical (should include itself) or not.

  If a term is an alt_id of something, or is obsolete (has a replaced_by
  relation), it is not an applicable label.

  Args:
    term: GoTerm.

  Returns:
    bool
  """
  is_replaced_by = any(relation_type == 'replaced_by'
                       for relation_type, _ in term.related_labels)
  alt_id = any(relation_type == _IS_NON_CANONICAL_ALT_ID_LABEL_OF
               for relation_type, _ in term.related_labels)
  return (not is_replaced_by) and (not alt_id)


def transitive_go_parenthood(go_terms: List[GoTerm]) -> Dict[Text, Set[Text]]:
  """Converts GoTerms (no transitive relations) to include transitive parents.

  Includes itself as one of its parents, with the exception of alt_ids and
  replaced_by tags.
  When a node has alt ids, its only parent is the term for which it is an
  alt_id. Note that a term may only be an alt_id for one term [1].
  When a node is obsolete, it has one or more replaced_by tags [2], and this
  obsolete name will not be included in the transitive right-hand-side.

  [1]
  https://github.com/geneontology/go-ontology/blob/7be0df46781f2e3a456a3e178def19dcdbb20ecf/src/util/check-obo-for-standard-release.pl#L134-L140
  [2] https://owlcollab.github.io/oboformat/doc/GO.format.obo-1_4.html

  Args:
    go_terms: List of GoTerm.

  Returns:
    Dict of term_id -> transitive set of parent term names.
  """
  go_term_dict = {
      t.term_id: frozenset(label for _, label in t.related_labels)
      for t in go_terms
  }

  transitive_go_terms = {
      t.term_id: _transitive_parenthood(t.term_id, go_term_dict)
      for t in tqdm.tqdm(go_terms, position=0)
  }

  terms_whose_labels_should_include_themselves = frozenset(
      term.term_id
      for term in go_terms
      if _go_term_applicable_labels_should_include_themselves(term))

  # Add in self to parents when it's not an alt_id or obsolete label. See
  # docstring for more information.
  for term_id in terms_whose_labels_should_include_themselves:
    transitive_go_terms[term_id].add(term_id)

  return transitive_go_terms


def _transitive_parenthood(key: Text,
                           term_dict: Dict[Text, FrozenSet[Text]]) -> Set[Text]:
  """Finds all parents, transitively, of `key` in `term_dict`.

  Does not include itself in the set of parents, regardless of the type of
  relation. This is left to the caller to decide.

  Args:
    key: Go Term, e.g. GO:0000001
    term_dict: Go term to set of parent go terms.

  Returns:
    Set of transitive parent go terms of `key`.
  """
  running_total = set()
  to_examine = set(term_dict[key])  # Make a copy so we can pop from it.

  while len(to_examine) > 0:  # pylint: disable=g-explicit-length-test
    cur_element = to_examine.pop()
    running_total.add(cur_element)

    for potential in term_dict[cur_element]:
      if potential not in running_total:
        to_examine.add(potential)

  return running_total


def _replace_one_level_up_with_dash_for_ec(s: Text) -> Text:
  """Finds direct parent of a label.

  Args:
    s: e.g. 1.2.3.4. Values including 'n' in one of their numbers [1] are
      treated like every other value. Non leaf nodes (those including a hyphen)
      are also allowed.

  Returns:
    E.g. 1.2.-.-
  """
  if s.count('-') == 0:
    return re.sub(EC_NUMBER_REGEX, '\\1.\\2.\\3.-', s)
  if s.count('-') == 1:
    return re.sub(EC_NUMBER_REGEX, '\\1.\\2.-.-', s)
  if s.count('-') == 2:
    return re.sub(EC_NUMBER_REGEX, '\\1.-.-.-', s)
  if s.count('-') == 3:
    return re.sub(EC_NUMBER_REGEX, '-.-.-.-', s)
  raise ValueError('Expected the number of hyphens in string to be between '
                   '0 and 3 (string was {}). Check that the input matches the '
                   'regex {}'.format(s, EC_NUMBER_REGEX))


def _all_ec_parents_for_label(label: Text) -> Set[Text]:
  """Computes all parents for an EC label.

  Does not include top level EC (level 0) value -.-.-.- in output.

  Args:
    label: e.g. 1.2.3.4. Values including 'n' in one of their numbers [1] are
      treated like every other value. Non leaf nodes (those including a hyphen)
      are also allowed.

  Returns:
    For e.g., 1.2.3.-, returns set(1.2.3.-, 1.2.-.-, 1.-.-.-)

    That is, this includes both the input `label`, as well as the root node
    -.-.-.-.
  """
  parent = label
  parents_set = set()
  while parent != _TOP_LEVEL_EC_CLASS_VALUE:  # Exclude -.-.-.- from output.
    parents_set.add(parent)  # First loop adds self to parenthood.
    parent = _replace_one_level_up_with_dash_for_ec(parent)

  return parents_set


def _get_leaf_node_ec_labels_from_file_contents(
    enzyme_dat_file_contents: Optional[Text] = None) -> List[Tuple[Text, Text]]:
  """Parses enzyme.dat file [1] into EC numbers and descriptions.

  [1] ftp://ftp.expasy.org/databases/enzyme/enzyme.dat
  [2] ftp://ftp.expasy.org/databases/enzyme/enzuser.txt

  Args:
    enzyme_dat_file_contents: Text of file at [1]. Follows format at [2].
      Contains only information about leaf nodes of the EC hierarchy (labels
      with no hyphens; e.g. 1.2.3.4). If None, the current file is parsed from
      CNS.

  Returns:
    List of string like "1.2.3.4", "oxalic acid oxidase".
    Note: ec numbers do not include the string "EC".
  """
  if enzyme_dat_file_contents is None:
    with open(EC_LEAF_NODE_METADATA_PATH) as f:
      enzyme_dat_file_contents = f.read()

  ids_and_descriptions = []

  # Beginning of EC file does not have to do with term parsing; we omit
  # the "0th" ID entry. See [1] in docstring for the format.
  id_blocks = enzyme_dat_file_contents.split('\nID')[1:]
  for block in id_blocks:
    lines_in_block = block.split('\n')
    term_id = re.findall(r'\s+(.*)', lines_in_block[0])[0]

    desc = ''
    for line in block.split('\n'):
      if line.startswith('DE'):
        desc += re.findall(r'DE\s+(.*)', line)[0]

    ids_and_descriptions.append((term_id, desc))

  return ids_and_descriptions


def _get_non_leaf_node_ec_labels_from_file_contents(
    enzyme_class_file_contents: Optional[Text] = None
) -> List[Tuple[Text, Text]]:
  """Parses enzclass.txt file [1] into EC numbers and descriptions.

  [1] ftp://ftp.expasy.org/databases/enzyme/enzclass.txt

  Args:
    enzyme_class_file_contents: Text of file at [1]. Contains only information
      about non-leaf nodes of the EC hierarchy (e.g. 1.2.3.-). If None, the
      current file is parsed from CNS.

  Returns:
    List of string like "1.-.-.-", "oxidoreductase".
    Note: ec numbers do not include the string "EC".
  """
  if enzyme_class_file_contents is None:
    with open(EC_NON_LEAF_NODE_METADATA_PATH) as f:
      enzyme_class_file_contents = f.read()

  non_leaf_node_label_lines = [
      l.strip()
      for l in enzyme_class_file_contents.split('\n')
      if _NON_LEAF_NODE_LINE_REGEX.match(l)
  ]

  terms_and_descriptions = []
  for line in non_leaf_node_label_lines:
    term_id = ''.join(line[0:9]).replace(' ', '')
    term_description = re.findall(r'.*.-\s+(.*)', line)[0]
    terms_and_descriptions.append((term_id, term_description))

  return terms_and_descriptions


def ec_label_to_description(
    enzyme_dat_file_contents: Optional[Text] = None,
    enzyme_class_file_contents: Optional[Text] = None) -> Dict[Text, Text]:
  """Get dictionary from EC label to description.

  [1] ftp://ftp.expasy.org/databases/enzyme/enzyme.dat
  [2] ftp://ftp.expasy.org/databases/enzyme/enzuser.txt
  [3] ftp://ftp.expasy.org/databases/enzyme/enzclass.txt

  Args:
    enzyme_dat_file_contents: Text of file at [1]. Follows format at [2].
      Contains only information about leaf nodes of the EC hierarchy (labels
      with no hyphens; e.g. 1.2.3.4).
    enzyme_class_file_contents: Text of file at [3]. Contains only information
      about non-leaf nodes of the EC hierarchy (e.g. 1.2.3.-).

  Returns:
    Dictionary from EC label to description. Non root-level terms have their
    parents information included for easier human-readability.
  """
  leaves = _get_leaf_node_ec_labels_from_file_contents(enzyme_dat_file_contents)
  non_leaves = _get_non_leaf_node_ec_labels_from_file_contents(
      enzyme_class_file_contents)

  term_to_description = {}
  for term, description in sorted(
      non_leaves + leaves, key=lambda x: x.count('-'), reverse=True):
    if term.count('-') == 3:
      term_to_description['EC:' + term] = description
    else:
      terms_parent = _replace_one_level_up_with_dash_for_ec(term)
      term_to_description[
          'EC:' +
          term] = term_to_description['EC:' + terms_parent] + ' ' + description

  return term_to_description


def parse_full_ec_file_to_transitive_parenthood(
    enzyme_dat_file_contents: Text,
    enzyme_class_file_contents: Text,
) -> Dict[Text, Set[Text]]:
  """Parses enzyme.dat [1] and enzclass.txt [3] into transitive parenthood dict.

  [1] ftp://ftp.expasy.org/databases/enzyme/enzyme.dat
  [2] ftp://ftp.expasy.org/databases/enzyme/enzuser.txt
  [3] ftp://ftp.expasy.org/databases/enzyme/enzclass.txt

  Args:
    enzyme_dat_file_contents: Text of file at [1]. Follows format at [2].
      Contains only information about leaf nodes of the EC hierarchy (labels
      with no hyphens; e.g. 1.2.3.4).
    enzyme_class_file_contents: Text of file at [3]. Contains only information
      about non-leaf nodes of the EC hierarchy (e.g. 1.2.3.-).

  Returns:
    Dict of all EC numbers to each of their parents. The values themselves
    are included in their parent set. Note that [1] only includes leaf nodes;
    this function includes all members of the tree as keys in the return value.

    Parents include a root node called -.-.-.- that indicates that this example
    is an enzyme.

    Output keys include the prefix "EC:".
  """
  leaf_node_labels = _get_leaf_node_ec_labels_from_file_contents(
      enzyme_dat_file_contents)
  non_leaf_node_labels = _get_non_leaf_node_ec_labels_from_file_contents(
      enzyme_class_file_contents)

  id_to_transitive_parents = {}
  for label, _ in tqdm.tqdm(leaf_node_labels + non_leaf_node_labels):
    parents_of_label = _all_ec_parents_for_label(label)

    # Also add parents themselves as labels in the dictionary.
    for parent_of_label in parents_of_label:
      if parent_of_label not in id_to_transitive_parents:
        rhs = set('EC:' + x for x in _all_ec_parents_for_label(parent_of_label))
        id_to_transitive_parents['EC:' + parent_of_label] = rhs

  return id_to_transitive_parents


def get_applicable_label_dict(
    path: Text = APPLICABLE_LABEL_JSON_PATH) -> Dict[Text, List[Text]]:
  with open(path, 'r') as f:
    with gzip.GzipFile(fileobj=f, mode='r') as gzip_file:
      return json.load(gzip_file)


def reverse_map(
    applicable_label_dict: Dict[Text, Collection[Text]],
    label_vocab: Optional[Set[Text]] = None) -> Dict[Text, FrozenSet[Text]]:
  """Flip parenthood dict to map parents to children.

  Args:
    applicable_label_dict: e.g. output of get_applicable_label_dict.
    label_vocab: e.g. output of inference_lib.vocab_from_model_base_path

  Returns:
    collections.defaultdict of k, v where:
    k: originally the values in applicable_label_dict
    v: originally the keys in applicable_label_dict.
    The defaultdict returns an empty frozenset for keys that are not found.
    This behavior is desirable for lifted clan label normalizers, where
    keys may not imply themselves.
  """
  # This is technically the entire transitive closure, so it is safe for DAGs
  # (e.g. GO labels).

  children = collections.defaultdict(set)
  for child, parents in applicable_label_dict.items():
    # Avoid adding children which don't appear in the vocab.
    if label_vocab is None or child in label_vocab:
      for parent in parents:
        children[parent].add(child)
  children = {k: frozenset(v) for k, v in children.items()}
  return collections.defaultdict(frozenset, children.items())


def is_implied_by_something_else(
    current_label: Text,
    reversed_normalizer: Dict[Text, FrozenSet[Text]],
    all_labels_for_protein: FrozenSet[Text],
) -> bool:
  """Returns whether the current label is implied by other labels for protein.

  Args:
    current_label: label about which we're asking "is this implied by some other
      label for this protein?"
    reversed_normalizer: output of reverse_map(label_normalizer). Helps this
      function run fast.
    all_labels_for_protein: set of all labels given to protein.

  Returns:
    bool
  """
  all_labels_for_protein_without_current = all_labels_for_protein - frozenset(
      [current_label])

  children_of_current_label = reversed_normalizer[current_label]

  # Most labels imply themselves; remove.
  children_of_current_label = children_of_current_label - frozenset(
      [current_label])

  return len(  # pylint: disable=g-explicit-length-test
      children_of_current_label.intersection(
          all_labels_for_protein_without_current)) > 0


def _filter_label_set_to_most_specific(
    label_set: FrozenSet[Text],
    reversed_normalizer: Dict[Text, FrozenSet[Text]]) -> FrozenSet[Text]:
  """Filters label set to most specific.

  Args:
    label_set: set of all labels given to protein.
    reversed_normalizer: output of reverse_map(label_normalizer). Helps this
      function run fast.

  Returns:
    Filtered set of labels.
  """
  return frozenset([
      l for l in label_set
      if not is_implied_by_something_else(l, reversed_normalizer, label_set)
  ])


def filter_labels_to_most_specific(
    df: pd.DataFrame,
    normalizer: Dict[Text, FrozenSet[Text]],
    column_to_filter: Text = 'predicted_label',
) -> pd.DataFrame:
  """Filter labels given to each protein to the most specific label.

  Useful for labels like GO, where we predict a ton of labels, and we only
  want to look at the most informative labels.

  Args:
    df: pd.DataFrame with column `column_to_filter`.
    normalizer: label normalizer.
    column_to_filter: name of column in df.

  Returns:
    pd.DataFrame with column `column_to_filter`.
  """
  reversed_normalizer = reverse_map(normalizer)

  working_df = df.copy()
  working_df[column_to_filter] = working_df[column_to_filter].apply(
      lambda label_set: _filter_label_set_to_most_specific(  # pylint: disable=g-long-lambda
          label_set, reversed_normalizer))
  return working_df
