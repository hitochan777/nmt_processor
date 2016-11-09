from unittest import TestCase
import os
import tempfile
from operator import itemgetter

from word_embedding import Word2Vec


class TestWord2Vec(TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.training_data = os.path.join(dir_path, "train10000.ja")

    def test_train(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        model_file.close()

    def test_set_vocab(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        w2v = Word2Vec(model_file.name)
        w2v.set_vocab(["誰", "私", "彼"])
        self.assertEqual(len(w2v.syn0norm_in_vocab), 3)
        model_file.close()

    def test_most_similar_word1(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        w2v = Word2Vec(model_file.name, topn=5)
        with self.assertRaisesRegex(AssertionError, r"^You need to call set_vocab first$"):
            _ = w2v.most_similar_word("人間")

        model_file.close()

    def test_most_similar_word2(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        w2v = Word2Vec(model_file.name, topn=5)
        w2v.set_vocab(["誰", "私", "彼"])
        most_sim_words = list(map(itemgetter(0), w2v.most_similar_word("人間")))
        self.assertEqual(len(most_sim_words), 3)
        self.assertIn("誰", most_sim_words)
        self.assertIn("私", most_sim_words)
        self.assertIn("彼", most_sim_words)
        model_file.close()

    def test_similarity(self):
        model_file = tempfile.NamedTemporaryFile()
        Word2Vec.train(model_file, self.training_data, size=100, window=5, negative=5, min_count=0, workers=4)
        w2v = Word2Vec(model_file.name, topn=5)
        sim = w2v.similarity("THIS_WORD_IS_NOT_IN_THE_TRAINING_DATA", "彼")
        self.assertAlmostEqual(sim, 0.0)
        sim = w2v.similarity("彼", "私")
        self.assertNotAlmostEquals(sim, 0.0)
        model_file.close()
