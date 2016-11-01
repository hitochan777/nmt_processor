from unittest import TestCase
import os
import tempfile

from word_embedding import Word2Vec


class TestWord2Vec(TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.training_data = os.path.join(dir_path, "train10000.ja")

    def test_train(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        model_file.close()

    def test_most_similar_word(self):
        self.fail()

    def test_set_vocab(self):
        self.fail()

    def test_similarity(self):
        self.fail()
