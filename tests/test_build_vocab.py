from unittest import TestCase
from io import StringIO
from build_vocab import build_vocab


class TestBuildVocab(TestCase):
    def test_build_vocab1(self):
        file_content = "あ い う え お\nあ い う え お\nか き く け こ\nさ し す せ そ"
        fs = StringIO(file_content)
        voc = build_vocab(fs)
        self.assertTrue(isinstance(voc, set))
        self.assertEqual(len(voc), 15)

    def test_build_vocab2(self):
        # test voc_limit option
        file_content = "あ い う え お\nあ い う え お\nか き く け こ\nさ し す せ そ"
        fs = StringIO(file_content)
        voc = build_vocab(fs, voc_limit=8)
        self.assertTrue(isinstance(voc, set))
        self.assertEqual(len(voc), 8)

    def test_build_vocab3(self):
        file_content = "あ い う え お\nあ い う え お\nか き く け こ\nさ し す せ そ"
        fs = StringIO(file_content)
        voc = build_vocab(fs, max_nb_ex=2)
        self.assertTrue(isinstance(voc, set))
        self.assertEqual(len(voc), 5)
        self.assertTrue("か" not in voc)
        self.assertTrue("き" not in voc)
