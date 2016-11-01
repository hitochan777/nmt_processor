from unittest import TestCase
from lexical_dictionary import LexicalDictionary
from io import StringIO
from operator import itemgetter


class TestLexicalDictionary(TestCase):
    def setUp(self):
        self.raw_data = (
            "私 I 0.4\n"
            "俺 I 0.1\n"
            "NULL I 0.3\n"
            "まじで really 0.3\n"
            "うち I 0.2\n"
            "本当に really 0.2\n"
            "本当 really 0.4\n"
            "実に really 0.1"
        )

    def test_create_lexical_table1(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=2)
        self.assertEqual(len(ld.entries), 2)
        self.assertTrue("I" in ld.entries)
        self.assertTrue("really" in ld.entries)

        self.assertTrue((0.4, "私") in ld.entries["I"])
        self.assertTrue((0.2, "うち") in ld.entries["I"])
        self.assertFalse((0.1, "俺") in ld.entries["I"])
        self.assertFalse((0.3, "NULL") in ld.entries["I"])

        self.assertTrue((0.4, "本当") in ld.entries["really"])
        self.assertTrue((0.3, "まじで") in ld.entries["really"])
        self.assertFalse((0.2, "本当に") in ld.entries["really"])
        self.assertFalse((0.1, "実に") in ld.entries["really"])

    def test_create_lexical_table2(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=None)
        self.assertEqual(len(ld.entries), 2)
        self.assertTrue("I" in ld.entries)
        self.assertTrue("really" in ld.entries)

        self.assertTrue((0.4, "私") in ld.entries["I"])
        self.assertTrue((0.2, "うち") in ld.entries["I"])
        self.assertTrue((0.1, "俺") in ld.entries["I"])
        self.assertFalse((0.3, "NULL") in ld.entries["I"])

        self.assertTrue((0.4, "本当") in ld.entries["really"])
        self.assertTrue((0.3, "まじで") in ld.entries["really"])
        self.assertTrue((0.2, "本当に") in ld.entries["really"])
        self.assertTrue((0.1, "実に") in ld.entries["really"])

    def test_get_translations(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=2)
        translations = ld.get_translations("I", only_in_vocab=False)
        self.assertListEqual(translations, [("私", 0.4), ("うち", 0.2)])
        translations = ld.get_translations("really", only_in_vocab=False)
        self.assertListEqual(translations, [("本当", 0.4), ("まじで", 0.3)])
        translations = ld.get_translations("unknown", only_in_vocab=False)
        self.assertListEqual(translations, [])

    def test_get_translations_in_vocab(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=2)
        ld.set_vocab(set(["うち", "俺", "本当"]))
        translations = ld.get_translations("I")
        self.assertListEqual(translations, [("うち", 0.2)])
        translations = ld.get_translations("really")
        self.assertListEqual(translations, [("本当", 0.4)])

    def test_filter_vocab1(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=2)
        ld.set_vocab(set(["私", "俺"]))
        filtered_vocab = ld.filter_vocab(set(["I", "really", "this"]))
        self.assertListEqual(filtered_vocab, ["I"])

    def test_filter_vocab2(self):
        fs = StringIO(self.raw_data)
        ld = LexicalDictionary.create_lexical_table(fs, topn=2)
        ld.set_vocab(set(["私", "俺", "本当"]))
        filtered_vocab = ld.filter_vocab(set(["I", "really", "this"]))
        self.assertEqual(set(filtered_vocab), set(["I", "really"]))
