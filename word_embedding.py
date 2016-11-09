import logging
import numpy as np
from gensim import matutils
from gensim.models import word2vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceGenerator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        if isinstance(self.filename, str):
            f = open(self.filename, "r")
        else:
            f = self.filename

        for line in f:
            yield line.rstrip().split(" ")


class Word2Vec(object):
    def __init__(self, model_path, topn=1):
        self.model_path = model_path
        self.topn = topn
        self.vocab = []
        self.model = word2vec.Word2Vec.load(model_path)
        logger.info("Finish loading word embedding model")

    @staticmethod
    def train(model_name, training_data_path, **kwargs):
        logger.info("Training the model")
        sentences = SentenceGenerator(training_data_path)
        model = word2vec.Word2Vec(sentences, **kwargs)
        logger.info("Saving the model")
        model.init_sims(replace=True)  # trim unneeded model memory = use (much) less RAM.
        model.save(model_name)
        logger.info("Trained model was saved to %s." % model_name)

    def most_similar_word(self, word):
        assert self.model is not None, "You have to load a model"
        assert self.vocab is not None, "You have to set vocab"
        assert self.topn is not None
        assert hasattr(self, "syn0norm_in_vocab"), "You need to call set_vocab first"
        topn = self.topn

        try:
            word_vec = self.model.syn0norm[self.model.vocab[word].index]
            dists = np.dot(self.syn0norm_in_vocab, word_vec)
            best_ids = matutils.argsort(dists, topn=topn+1, reverse=True)
            result = [(self.vocab[best_id], float(dists[best_id])) for best_id in best_ids if self.vocab[best_id] != word]
            assert all(vocab in self.vocab for vocab, dist in result)
            return result[:topn]

        except KeyError:
            logger.info("%s not found in word2vec model" % word)
            return [(word, 1.0)]

    def set_vocab(self, vocab):
        assert self.model is not None
        if type(vocab) is set:
            vocab = list(vocab)

        self.vocab = vocab
        self.model.init_sims()
        indices = list(map(lambda word: self.model.vocab[word].index, vocab))
        self.syn0norm_in_vocab = self.model.syn0norm[indices]

    def similarity(self, w1, w2):
        try:
            return self.model.similarity(w1, w2)
        except KeyError:
            return 0.0
