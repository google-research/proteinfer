import gzip
import xml.etree.cElementTree
import protein_dataset
import utils
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import json
import gzip
from absl import flags
from absl import app
import tqdm
import os


"""
Generate a UniProt dataset.

Usage:

wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz
wget ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping_selected.tab.gz
python generate_dataset.py --id_mapping_file=idmapping_selected.tab.gz --uniprot_xml=uniprot_sprot.xml.gz --parenthood_file=./testdata/parenthood.json.gz --dataset_type=clustered --output_prefix=clustered
python generate_dataset.py --uniprot_xml=uniprot_sprot.xml.gz --parenthood_file=./testdata/parenthood.json.gz --dataset_type=random --output_prefix=random
python generate_dataset.py --tfrecord_files=random*.tfrecord --vocab_prefix=EC --dataset_type=vocab --output_prefix=./vocabs


"""

FOLDS = np.array([
    protein_dataset.TRAIN_FOLD, protein_dataset.DEV_FOLD,
    protein_dataset.TEST_FOLD
])
FOLD_PROBABILITIES = np.array([0.8, 0.1, 0.1])
RANDOM_SEED = 8

NAMESPACE = "{http://uniprot.org/uniprot}"
RELEVANT_LABEL_TYPES = ["GO", "EC", "Pfam"]

flags.DEFINE_string('parenthood_file', None, 'Gzipped parenthood file')

flags.DEFINE_string('uniprot_xml', None, 'Gzipped UniProtXML file')

flags.DEFINE_string('output_prefix', None, 'Output location for TF records')

flags.DEFINE_string('id_mapping_file', None,
                    'Mapping from UniRef clusters to members')

flags.DEFINE_string('dataset_type', "random",
                    'Type of dataset, "random" or "clustered"')


flags.DEFINE_string('tfrecord_files', None,
                    '')
flags.DEFINE_string('vocab_prefix', None,
                    '')

FLAGS = flags.FLAGS


class LabelParenthoodAdder():
    label_parenthood_mapping = None

    def __init__(self, parenthood_file):
        self.label_parenthood_mapping = json.load(gzip.open(parenthood_file))

    def add_parenthood(self, list_of_labels):
        label_set = set(list_of_labels)
        for label in list_of_labels:
            try:
                label_set.update(self.label_parenthood_mapping[label])
            except KeyError:
                # This should only happen when the dataset is more up to date
                # than the parenthood file and so some new terms have been added
                pass
        return label_set


class ClusteredDatasetSampler():
    uniprot_to_uniref = {}
    uniref_to_fold = {}

    def __init__(self, xml_filename, id_mapping_file):
        """
        Helper to use UniRef clusters to create a clustered dataset.

        Args:
            xml_filename: gzipped UniProt XML file
            id_mapping_file: gzipped UniProt ID mapping file
        """
        self.uniprot_to_uniref = self._generate_uniref_mapping(
            xml_filename, id_mapping_file)
        unirefs = {
            uniref
            for uniprot, uniref in self.uniprot_to_uniref.items()
        }
        self.uniref_to_fold = {uniref: sample_fold() for uniref in unirefs}

    def _generate_uniref_mapping(self, xml_filename, id_mapping):
        """Create a UniProt:UniRef dictionary from a UniRef id_mapping file and a UniProt XML file.
        
        The UniProt XML file is used to specify the subset of UniProt accessions we wish to extract."""
        dict_source = yield_dicts_from_xml_file(gzip.open(xml_filename))
        ids_in_dataset = set()
        accession_to_uniref_mapping = {}
        for example_dict in tqdm.tqdm(dict_source,position=0):
            ids_in_dataset.add(example_dict[protein_dataset.SEQUENCE_ID_KEY])
        for line in tqdm.tqdm(gzip.open(id_mapping, "rt"),position=0):
            items = line.split("\t")
            accession = items[0]
            if accession in ids_in_dataset:
                uniref50 = items[9]
                if uniref50.strip() != "":
                    accession_to_uniref_mapping[accession] = uniref50
        return accession_to_uniref_mapping

    def get_fold(self, seq_id):
        return self.uniref_to_fold[self.uniprot_to_uniref[seq_id]]


def sample_fold():
    return np.random.choice(FOLDS, p=FOLD_PROBABILITIES)

def _contains_non_standard_amino_acid(seq):
    for c in seq:
        if c not in utils.AMINO_ACID_VOCABULARY:
            return True
    return False
def yield_dicts_from_xml_file(source):
    """Yield example dicts from a UniProt format XML file."""
    with tqdm.tqdm(position=0) as pbar:
        for _, elem in xml.etree.cElementTree.iterparse(source):
            if elem.tag == f"{NAMESPACE}entry":
                pbar.update(1)
                accession = elem.find(f"{NAMESPACE}accession").text
                sequence = elem.find(f"{NAMESPACE}sequence").text
                db_refs = elem.findall(f"{NAMESPACE}dbReference")
                labels = []
                for db_ref in db_refs:
                    db_ref_id = db_ref.attrib['id']
                    db_ref_type = db_ref.attrib['type']
                    if db_ref_type in RELEVANT_LABEL_TYPES:
                        if db_ref_type == "GO":
                            db_ref_id = db_ref_id.replace("GO:",
                                                          "")  #avoid GO:GO:
                        labels.append(f"{db_ref_type}:{db_ref_id}")
                if not 'fragment' in elem.find(f"{NAMESPACE}sequence").attrib.keys() and not _contains_non_standard_amino_acid(sequence):
                                 yield ({
                                        protein_dataset.SEQUENCE_ID_KEY: accession,
                                        protein_dataset.SEQUENCE_KEY: sequence,
                                        protein_dataset.LABEL_KEY: labels
                                 })
                elem.clear()


def proto_from_dict(input_dict):
    """Create a TF Example from an example dictionary."""
    tf_labels = [str.encode(x) for x in input_dict[protein_dataset.LABEL_KEY]]
    features = {
        protein_dataset.SEQUENCE_ID_KEY:
        tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[str.encode(input_dict[protein_dataset.SEQUENCE_ID_KEY])])),
        protein_dataset.SEQUENCE_KEY:
        tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[str.encode(input_dict[protein_dataset.SEQUENCE_KEY])])),
        protein_dataset.LABEL_KEY:
        tf.train.Feature(bytes_list=tf.train.BytesList(value=tf_labels))
    }
    example_proto = tf.train.Example(features=tf.train.Features(
        feature=features))
    return example_proto


def create_random_dataset(xml_filename, tfrecord_prefix,
                          label_parenthood_file):
    """Create TFRecords of TFexamples from a gzipped UP XML file."""
    label_parenthood_adder = LabelParenthoodAdder(label_parenthood_file)
    dict_source = yield_dicts_from_xml_file(gzip.open(xml_filename))
    writers = {}
    for fold in FOLDS:
        writers[fold] = tf.io.TFRecordWriter(
            f'{tfrecord_prefix}_{fold}.tfrecord')
    np.random.seed(8)
    for example_dict in dict_source:
        example_dict[
            protein_dataset.LABEL_KEY] = label_parenthood_adder.add_parenthood(
                example_dict[protein_dataset.LABEL_KEY])
        example_dict[protein_dataset.LABEL_KEY] = sorted(
            example_dict[protein_dataset.LABEL_KEY])
        writers[sample_fold()].write(
            proto_from_dict(example_dict).SerializeToString())


def create_clustered_dataset(xml_filename, tfrecord_prefix, id_mapping_file,
                             label_parenthood_file):
    """Create TFRecords of TFexamples from a gzipped UniProt XML file."""

    clustered_dataset_sampler = ClusteredDatasetSampler(
        xml_filename, id_mapping_file)
    #raise ValueError(clustered_dataset_sampler.uniprot_to_uniref, clustered_dataset_sampler.uniref_to_fold)
    label_parenthood_adder = LabelParenthoodAdder(label_parenthood_file)

    dict_source = yield_dicts_from_xml_file(gzip.open(xml_filename))
    writers = {}
    for fold in FOLDS:
        writers[fold] = tf.io.TFRecordWriter(
            f'{tfrecord_prefix}_{fold}.tfrecord')

    np.random.seed(RANDOM_SEED)

    for example_dict in dict_source:
        example_dict[
            protein_dataset.LABEL_KEY] = label_parenthood_adder.add_parenthood(
                example_dict[protein_dataset.LABEL_KEY])
        fold = None
        try:
            fold = clustered_dataset_sampler.get_fold(
            example_dict[protein_dataset.SEQUENCE_ID_KEY])
        except KeyError:
            print(f"No fold found for {example_dict[protein_dataset.SEQUENCE_ID_KEY]}")
        if fold:
            writers[fold].write(proto_from_dict(example_dict).SerializeToString())

def create_vocab(tfrecord_files, prefix, output_prefix):
    """Generate a vocabulary from TFrecord files."""
    sequence_iterator = protein_dataset.yield_examples(tfrecord_files)
    label_set = set()
    for example in tqdm.tqdm(sequence_iterator):
        label_set.update({x.decode("utf-8") for x in example[protein_dataset.LABEL_KEY] if x.decode("utf-8").startswith(prefix)})

    vocab = pd.DataFrame({'vocab_item':sorted(label_set)})
    vocab['vocab_index'] = range(vocab.shape[0])
    path_loc = os.path.join(output_prefix,f"{prefix}.tsv")
    vocab.to_csv(path_loc,sep="\t")

def main(_):
    if FLAGS.dataset_type == "random":
        create_random_dataset(FLAGS.uniprot_xml, FLAGS.output_prefix,
                              FLAGS.parenthood_file)
    elif FLAGS.dataset_type == "clustered":
        create_clustered_dataset(FLAGS.uniprot_xml, FLAGS.output_prefix,
                                 FLAGS.id_mapping_file, FLAGS.parenthood_file)
    elif FLAGS.dataset_type == "vocab":
        create_vocab(FLAGS.tfrecord_files, FLAGS.vocab_prefix, FLAGS.output_prefix)
    else:
        raise ValueError("Invalid dataset_type")


if __name__ == '__main__':
    FLAGS.alsologtostderr = True

    app.run(main)
