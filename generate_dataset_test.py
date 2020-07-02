import generate_dataset
import protein_dataset
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import gzip
import tempfile
import os


class GenerateDatasetTest(parameterized.TestCase):
    def test_yield_dicts_from_xml_file(self):
        yielder = generate_dataset.yield_dicts_from_xml_file(
            gzip.open("./testdata/camel_test.xml.gz", ))
        actual = list(yielder)
        expected = [{
            'id':
            'P67929',
            'sequence':
            'SETAAEKFERQHMDSYSSSSSNSNYCNQMMKRREMTDGWCKPVNTFIHESLEDVQAVCSQKSVTCKNGQTNCHQSSTTMHITDCRETGSSKYPNCAYKASNLQKHIIIACEGNPYVPVHFDASV',
            'label': [
                'EC:4.6.1.18', 'GO:0005576', 'GO:0016829', 'GO:0003676',
                'GO:0004522', 'Pfam:PF00074'
            ]
        }, {
            'id':
            'P63105',
            'sequence':
            'VLSSKDKTNVKTAFGKIGGHAAEYGAEALERMFLGFPTTKTYFPHFDLSHGSAQVKAHGKKVGDALTKAADHLDDLPSALSALSDLHAHKLRVDPVNFKLLSHCLLVTVAAHHPGDFTPSVHASLDKFLANVSTVLTSKYR',
            'label': [
                'GO:0005833', 'GO:0020037', 'GO:0005506', 'GO:0019825',
                'GO:0005344', 'Pfam:PF00042'
            ]
        }, {
            'id':
            'Q865W7',
            'sequence':
            'MNSLSTSAFSPVAFSLGLLLVMATAFPTPVPLGEDFKDGTTSNRPFTSPDKSEELIKYILGRISAMRKEMCEKYDKCENSKEALSENNLNLPKMTEKDGCFQSGFNQETCLMRITIGLLEFQIYLDYLQNYYEGDKGNTEAVQISTKALIQLLRQKVKQPEEVSTPNPITGSSLLNKLQTENQWMKNTKMILILRSLEDFLQFSLRAVRIM',
            'label': [
                'GO:0005615', 'GO:0005125', 'GO:0008083', 'GO:0005138',
                'GO:0006953', 'GO:0072540', 'Pfam:PF00489'
            ]
        }, {
            'id':
            'P86314',
            'sequence':
            'YLDHGLGAPAPYVDPLEPKREVCELNPDCDELADQMGFQEAYRRFYGTT',
            'label': [
                'GO:0005737', 'GO:0005576', 'GO:0005509', 'GO:0031214',
                'GO:0060348', 'GO:0030500', 'GO:1900076', 'GO:0032571'
            ]
        }, {
            'id':
            'P68230',
            'sequence':
            'MVHLSGDEKNAVHGLWSKVKVDEVGGEALGRLLVVYPWTRRFFESFGDLSTADAVMNNPKVKAHGSKVLNSFGDGLNHLDNLKGTYAKLSELHCDKLHVDPENFRLLGNVLVVVLARHFGKEFTPDLQAAYQKVVAGVANALAHRYH',
            'label': [
                'GO:0005833', 'GO:0020037', 'GO:0046872', 'GO:0019825',
                'GO:0005344', 'Pfam:PF00042'
            ]
        }, {
            'id':
            'Q865W6',
            'sequence':
            'MNYTSYILAFQLCVILGSSGCYCQAPFFDEIENLKKYFNASNPDVADGGPLFLEILKNWKEESDKKIIQSQIVSFYFKLFENLKDNQIIQRSMDIIKQDMFQKFLNGSSEKLEDFKKLIQIPVDNLKVQRKAISELIKVMNDLSPKSNLRKRKRSQNLFRGRRASK',
            'label': [
                'GO:0005615', 'GO:0005125', 'GO:0005133', 'GO:0051607',
                'GO:0006955', 'GO:0040008', 'Pfam:PF00714'
            ]
        }, {
            'id':
            'Q2PE47',
            'sequence':
            'MYKLQFLSCIALTLALVANSAPTLSSTKDTKKQLEPLLLDLQFLLKEVNNYENLKLSRMLTFKFYMPKKATELKHLQCLMEELKPLEEVLNLAQSKNSHLTNIKDSMNNINLTVSELKGSETGFTCEYDDETVTVVEFLNKWITFCQSIYSTLT',
            'label': [
                'GO:0005615', 'GO:0005125', 'GO:0008083', 'GO:0005134',
                'GO:0002250', 'Pfam:PF00715'
            ]
        }, {
            'id':
            'A5Z1X6',
            'sequence':
            'MNLQLIFWIGLISSVCCVFGQADEDRCLKANAKSCGECIQAGPNCGWCTNSTFLQEGMPTSARCDDLEALKKKGCHPNDTENPRGSKDIKKNKNVTNRSKGTAEKLQPEDITQIQPQQLVLQLRSGEPQTFTLKFKRAEDYPIDLYYLMDLSYSMKDDLENVKSLGTDLMNEMRRITSDFRIGFGSFVEKTVMPYISTTPAKLRNPCTNEQNCTSPFSYKNVLSLTDKGEVFNELVGKQRISGNLDSPEGGFDAIMQVAVCGSLIGWRNVTRLLVFSTDAGFHFAGDGKLGGIVLPNDGQCHLKNDVYTMSHYYDYPSIAHLVQKLSENNIQTIFAVTEEFQPVYKELKNLIPKSAVGTLSANSSNVIQLIIDAYNSLSSEVILENSKLPEGVTINYKSYCKNGVNGTGENGRKCSNISIGDEVQFEISITANKCPDKNSETIKIKPLGFTEEVEIILQFICECECQGEGIPGSPKCHDGNGTFECGACRCNEGRVGRHCECSTDEVNSEDMDAYCRKENSSEICSNNGECVCGQCVCRKRDNTNEIYSGKFCECDNFNCDRSNGLICGGNGVCKCRVCECNPNYTGSACDCSLDTTSCMAVNGQICNGRGVCECGACKCTDPKFQGPTCEMCQTCLGVCAEHKECVQCRAFNKGEKKDTCAQECSHFNITKVENRDKLPQPGQVDPLSHCKEKDVDDCWFYFTYSVNGNNEATVHVVETPECPTGPDIIPIVAGVVAGIVLIGLALLLIWKLLMIIHDRREFAKFEKERMNAKWDTGENPIYKSAVTTVVNPKYEGK',
            'label': [
                'GO:0009986', 'GO:0005925', 'GO:0008305', 'GO:0071438',
                'GO:0030027', 'GO:0042470', 'GO:0016020', 'GO:0055037',
                'GO:0032587', 'GO:0046872', 'GO:0046982', 'GO:0038023',
                'GO:0033627', 'GO:0007160', 'GO:0071404', 'GO:0007229',
                'GO:0030335', 'GO:1903078', 'GO:0031623', 'GO:0010710',
                'Pfam:PF07974', 'Pfam:PF18372', 'Pfam:PF08725', 'Pfam:PF07965',
                'Pfam:PF00362', 'Pfam:PF17205'
            ]
        }, {
            'id':
            'Q865W5',
            'sequence':
            'MALWLTVVIAFTCIGGLASPVPTPSPKALKELIEELVNITQNQKAPLCNGSMVWSINLTTSMYCAARESLINITNCSVIQRTQRMLNALCPHKLSAKVSSEHVRDTKIEVTQFIKTLLQHSRNVFHYRSFNWSKKS',
            'label': [
                'GO:0005615', 'GO:0005125', 'GO:0005126', 'GO:0006955',
                'Pfam:PF03487'
            ]
        }, {
            'id':
            'Q34028',
            'sequence':
            'MTNIRKSHPLLKIMNDAFIDLPAPSNISSWWNFGSLLGVCLIMQILTGLFLAMHYTSDTTTAFSSVAHICRDVNYGWIIRYLHANGASMFFICLYIHVGRGLYYGSYTFLETWNVGIILLFTVMATAFMGYVLPWGQMSFWGATVITNLLSAIPYIGTTLVEWIWGGFSVDKATLTRFFAFHFILPFIITALVAVHLLFLHETGSNNPTGISSDMDKIPFHPYYTIKDILGALLLMLILLILVLFSPDLLGDPDNYTPANPLNTPPHIKPEWYFLFAYAILRSIPNKLGGVLALILSILILALIPMLHTSKQRSMMFRPISQCLFWVLVADLLTLTWIGGQPVEPPFIMIGQVASILYFSLILILMPVAGIIENRILKW',
            'label': [
                'GO:0016021', 'GO:0005743', 'GO:0045275', 'GO:0046872',
                'GO:0008121', 'GO:0006122', 'Pfam:PF00032', 'Pfam:PF00033'
            ]
        }, {
            'id':
            'Q75N23',
            'sequence':
            'MSTESMIRDVELAEEALPKKAGGPQGSRRCLCLSLFSFLLVAGATTLFCLLHFGVIGPQKEELLTGLQLMNPLAQTLRSSSQASRDKPVAHVVADPAAQGQLQWEKRFANTLLANGVKLEDNQLVVPTDGLYLIYSQVLFSGQRCPSTPVFLTHTISRLAVSYPNKANLLSAIKSPCQGGTSEEAEAKPWYEPIYLGGVFQLEKDDRLSAEINMPNYLDFAESGQVYFGIIAL',
            'label': [
                'GO:0005615', 'GO:0016021', 'GO:0005886', 'GO:0005125',
                'GO:0005164', 'GO:0006955', 'GO:0097527', 'GO:0043242',
                'GO:0043065', 'GO:0043507', 'GO:0043406', 'GO:0051092',
                'GO:0001934', 'GO:0043243', 'Pfam:PF00229'
            ]
        }]
        self.assertEqual(actual, expected)

    def test_create_random_dataset(self):
        # includes coverage for proto_from_dict

        temp_dir = tempfile.TemporaryDirectory()
        tfrecord_prefix = os.path.join(temp_dir.name, "test_random")
        generate_dataset.create_random_dataset(
            "./testdata/camel_test.xml.gz", tfrecord_prefix,
            "./testdata/parenthood.json.gz")
        actual = list(
            protein_dataset.yield_examples(
                f"{tfrecord_prefix}_train.tfrecord"))[5]
        expected = {
            'label': [
                b'GO:0002376', b'GO:0003674', b'GO:0005102', b'GO:0005125',
                b'GO:0005126', b'GO:0005488', b'GO:0005515', b'GO:0005575',
                b'GO:0005576', b'GO:0005615', b'GO:0006955', b'GO:0007154',
                b'GO:0007165', b'GO:0008150', b'GO:0009987', b'GO:0023052',
                b'GO:0030545', b'GO:0030546', b'GO:0044421', b'GO:0048018',
                b'GO:0050789', b'GO:0050794', b'GO:0050896', b'GO:0051716',
                b'GO:0065007', b'GO:0065009', b'GO:0098772', b'Pfam:CL0053',
                b'Pfam:PF03487'
            ],
            'id':
            b'Q865W5',
            'sequence':
            b'MALWLTVVIAFTCIGGLASPVPTPSPKALKELIEELVNITQNQKAPLCNGSMVWSINLTTSMYCAARESLINITNCSVIQRTQRMLNALCPHKLSAKVSSEHVRDTKIEVTQFIKTLLQHSRNVFHYRSFNWSKKS'
        }
        self.assertEqual(actual, expected)

    def test_create_clustered_dataset(self):
        temp_dir = tempfile.TemporaryDirectory()
        tfrecord_prefix = os.path.join(temp_dir.name, "test_random")
        generate_dataset.create_clustered_dataset(
            "./testdata/camel_test.xml.gz", tfrecord_prefix,
            "./testdata/camel_id_mapping.tab.gz",
            "./testdata/parenthood.json.gz")
        actual = {
            x[protein_dataset.SEQUENCE_ID_KEY]
            for x in protein_dataset.yield_examples(
                f"{tfrecord_prefix}_train.tfrecord")
        }

        cluster = [b'Q865W6', b'Q865W5', b'P68230', b'P63105', b'Q34028']
        for x in cluster:
            assert (x in actual)


if __name__ == '__main__':
    absltest.main()
