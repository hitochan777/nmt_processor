from enum import Enum
import os
import json
import logging

from dotmap import DotMap

from word_embedding import Word2Vec
from lexical_dictionary import LexicalDictionary
from build_vocab import build_vocab


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ReplaceMethod = Enum("ReplaceMethod", "simple bilm bisim copy")


class Utility(object):
    @staticmethod
    def get_one_to_one_links(src, tgt, align, src_voc, tgt_voc):
        assert align is not None
        assert isinstance(src_voc, set), type(src_voc)
        assert isinstance(tgt_voc, set), type(tgt_voc)

        logged_pair = False

        # if the count is more than one, it's not one-to-one
        count_f = [0] * len(src)
        count_e = [0] * len(tgt)

        for fIndex, eIndices in align.items():
            for eIndex in eIndices:
                count_f[fIndex] += 1
                count_e[eIndex] += 1

        for fIndex, eIndices in align.items():
            if count_f[fIndex] > 1:
                continue

            is_src_unk = src[fIndex] not in src_voc
            for eIndex in eIndices:
                if count_e[eIndex] > 1:
                    continue

                is_tgt_unk = tgt[eIndex] not in tgt_voc
                if is_src_unk or is_tgt_unk:
                    if not logged_pair:
                        logger.debug("\nsrc: %s\ntgt: %s" % (" ".join(src), " ".join(tgt)))
                        logged_pair = True

                    logger.debug("%s-%s in (%s, %s)" % ("unk" if is_src_unk else "common", "unk" if is_tgt_unk else "common", src[fIndex], tgt[eIndex]))

                yield (fIndex, is_src_unk), (eIndex, is_tgt_unk)


class Replacer(object):
    def __init__(self, **kwargs):
        self.setting = DotMap(kwargs)
        self.src_voc, self.tgt_voc = self._load_vocab()
        self.src_embedding = self._load_src_embedding()
        self.tgt_embedding = self._load_tgt_embedding()
        self.lexe2f, self.lexf2e = self._load_lex_tables()

        if ReplaceMethod[self.setting.mode] in [ReplaceMethod.simple, ReplaceMethod.bilm, ReplaceMethod.bisim]:
            self.lexf2e.set_vocab(self.tgt_voc)
            if self.setting.guarantee_in_vocab_replace:
                filtered_src_voc = self.lexf2e.filter_vocab(self.src_voc)
                logger.info("The size of filtered source vocab: %d" % len(filtered_src_voc))
                self.src_embedding.set_vocab(filtered_src_voc) # set filtered_src_voc to src_embedding also when processing test data
            else:
                self.src_embedding.set_vocab(self.src_voc)

    def _should_load_src_embedding(self):
        if ReplaceMethod[self.setting.mode] in [ReplaceMethod.simple, ReplaceMethod.bilm, ReplaceMethod.bisim]:
            assert self.setting.src_embedding is not None
            return True

        return False

    def _load_src_embedding(self):
        if self._should_load_src_embedding():
             return Word2Vec(model_path=self.setting.src_embedding, topn=self.setting.src_embedding_topn)
        else:
            return None

    def _should_load_tgt_embedding(self):
        if (ReplaceMethod[self.setting.mode] == ReplaceMethod.bisim
                or self.setting.tgt_sim_threshold is not None):
            return True

        return False

    def _load_tgt_embedding(self):
        if self._should_load_tgt_embedding():
            return Word2Vec(model_path=self.setting.tgt_embedding)
        else:
            return None

    def _load_lex_tables(self):
        if ReplaceMethod[self.setting.mode] == ReplaceMethod.copy:
            lexe2f = None
        else:
            lexe2f = LexicalDictionary.create_lexical_table(self.setting.dictionary + ".e2f", topn=self.setting.dictionary_topn)

        lexf2e = LexicalDictionary.create_lexical_table(self.setting.dictionary + ".f2e", topn=self.setting.dictionary_topn)
        return lexe2f, lexf2e

    def _load_vocab(self):
        if self.setting.vocab_file is not None and os.path.isfile(self.setting.vocab_file):
            logger.info("Loading vocabularies from %s" % self.setting.vocab_file)
            with open(self.setting.vocab_file, "r") as f:
                vocab_data = json.load(f)
                assert len(vocab_data) == 2
                src_vcb = set(vocab_data[0])
                tgt_vcb = set(vocab_data[1])

            assert self.setting.src_voc_size is None or len(src_vcb) == self.setting.src_voc_size
            assert self.setting.tgt_voc_size is None or len(tgt_vcb) == self.setting.tgt_voc_size

            if self.setting.only_test:
                return src_vcb, None
            else:
                return src_vcb, tgt_vcb

        logger.info("Building vocabularies from training data")
        assert self.setting.src_file_for_voc is not None and self.setting.tgt_file_for_voc is not None, "Both source and target file is needed to build vocabulary"
        src_vcb = build_vocab(self.setting.src_file_for_voc, voc_limit=self.setting.src_voc_size)
        tgt_vcb = build_vocab(self.setting.tgt_file_for_voc, voc_limit=self.setting.tgt_voc_size)

        assert len(src_vcb) == self.setting.src_voc_size
        assert len(tgt_vcb) == self.setting.tgt_voc_size

        if self.setting.vocab_file is not None:
            logger.info("Writing vocab to %s" % self.setting.vocab_file)
            with open(self.setting.vocab_file, "w"):
                json.dump([list(src_vcb), list(tgt_vcb)])

            logger.info("Finished writing vocab to file")

        return src_vcb, tgt_vcb

    def replace(self, src, tgt=None, align=None):
        """
        – unk to unk, both the source and target word in the aligned
          pair are rare words. In this case we will replace the
          source word with a similar in-vocabulary word and the
          target word with the translation of the similar word.
        – unk to common, only the source word is rare. In this
          case we will keep the target word and replace the source
          word with the translation of the target word.
        – common to unk, only the target word is rare. In this case
          we will keep the source word and replace the target word
          with the translation of the source word.
        – common to common, no replacement in this case.
        – unk to null or null to unk, source or target rare word is
          not aligned to any word. In this case we simply remove
          the rare word from the sentence.
        """

        if tgt is not None:
            assert align is not None
            assert isinstance(self.src_voc, set), type(self.src_voc)
            assert isinstance(self.tgt_voc, set), type(self.tgt_voc)

            fwords = list(src)
            ewords = list(tgt)
            one_to_one_links = Utility.get_one_to_one_links(src, tgt, align, self.src_voc, self.tgt_voc)
            for (fIndex, is_src_unk), (eIndex, is_tgt_unk) in one_to_one_links:
                if is_src_unk:
                    if is_tgt_unk:
                        fwords[fIndex], ewords[eIndex] = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                    else:
                        fwords[fIndex], ewords[eIndex] = self.get_new_src(fIndex, eIndex, src, tgt)
                else:
                    if is_tgt_unk:
                        fwords[fIndex], ewords[eIndex] = self.get_new_tgt(fIndex, eIndex, src, tgt)
                    else:
                        fwords[fIndex] = src[fIndex]
                        ewords[eIndex] = tgt[eIndex]

            fwords = filter(None, fwords)
            ewords = filter(None, ewords)
            return fwords, ewords
        else:
            assert self.src_embedding is not None
            assert isinstance(self.src_voc, set)

            fwords = []

            for fIndex, fword in enumerate(src):
                if fword not in self.src_voc:
                    best_fword = self.get_src_substitution(src, fIndex)
                    fwords.append(best_fword)
                else:
                    fwords.append(fword)

            return fwords

    def get_new_src_tgt(self, fIndex, eIndex, src, tgt):
        candidates = []
        most_sim_words = self.src_embedding.most_similar_word(src[fIndex])

        if self.replace_both and src[fIndex] in self.src_dic:
            most_sim_words.insert(0, (src[fIndex], 1.0))

        logger.debug("most similar words of %s in the src vocab are %s" % (src[fIndex], str(most_sim_words)))
        if self.src_sim_threshold is not None and len(most_sim_words) > 0 and most_sim_words[0][1] < self.src_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_sim_threshold,))
            most_sim_words = []

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src[fIndex] or most_sim_word in self.src_dic, "%s is not in the dictionary!" % most_sim_word
            translations = self.lexf2e.get_translations(most_sim_word, only_in_vocab=True)
            if len(translations) > 0:
                candidates += list(zip_longest([(most_sim_word, cos_sim)], translations, fillvalue=(most_sim_word, cos_sim)))

        logger.debug("Candidates are %s" % (str(candidates),))

        if len(candidates) > 0:
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            new_fword = best_fword
            new_eword = best_eword
            if self.tgt_sim_threshold is not None:
                valid = self._is_tgt_replace_valid(tgt[eIndex], new_eword)
                if not valid:
                    logger.debug("Similarity of e(%s) and e'(%s) is less than %f, so not replacing." % (tgt[eIndex], new_eword, self.tgt_sim_threshold,))
                    if self.setting.backoff_to_unk:
                        new_fword = new_eword = self.setting.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        new_fword = src[fIndex]
                        new_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))

            assert (src[fIndex] == best_fword or (best_fword in self.src_dic)), "%s->%s" % (src[fIndex], best_fword)
        else:
            if self.setting.backoff_to_unk:
                new_fword = new_eword = self.setting.unk_tag
                logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
            else:
                logger.debug("No replacement because of no candidates")
                new_fword = src[fIndex]
                new_eword = tgt[eIndex]

        return new_fword, new_eword

    def get_new_src(self, fIndex, eIndex, src, tgt):
        if self.replace_both:
            return self.get_new_src_tgt(fIndex, eIndex, src, tgt)

        translations = self.lexe2f.get_translations(tgt[eIndex], only_in_vocab=True)

        if self.lex_prob_threshold is not None and len(translations) > 0 and translations[0][1] < self.lex_prob_threshold:
            logger.debug("Translations for %s are invalidated because the lexical translation probability(%f) is less than %f" % (tgt[eIndex], translations[0][1], self.lex_prob_threshold))
            translations = []

        if len(translations) > 0:
            probs = [translation[1] for translation in translations]
            tgt_candidates = list(zip_longest([tgt[eIndex]], probs ,fillvalue=tgt[eIndex]))
            for i in range(len(translations)):
                translations[i] = (translations[i][0], self.src_embedding.similarity(translations[i][0], src[fIndex]))
            candidates = list(zip_longest(translations, tgt_candidates))
            logger.debug("Candidates are %s" % (str(candidates),))
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            if self.src_sim_threshold is not None:
                try:
                    similarity = self.src_embedding.similarity(src[fIndex], best_fword)
                except KeyError:
                    logger.debug("Either %s or %s are not in the source vocab" % (src[fIndex], best_fword))
                    similarity = 0.0

                if similarity < self.setting.src_sim_threshold:
                    logger.debug("Similarity(%f) between %s and %s is less than %f, so not replacing." % (similarity, src[fIndex], best_fword, self.src_sim_threshold,))
                    if self.setting.backoff_to_unk:
                        best_fword = best_eword = self.setting.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        best_fword = src[fIndex]
                        best_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))
            else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))
            return best_fword, best_eword
        else:
            if self.setting.guarantee_in_vocab_replace:
                logger.debug("Trying to replace both src and tgt...")
                new_fword, new_eword = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                # assert new_fword in self.src_dic and new_eword in self.tgt_dic
                return new_fword, new_eword
            else:
                if self.backoff_to_unk:
                    new_fword = new_eword = self.unk_tag
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                else:
                    logger.debug("No replacement because of no candidates")
                    new_fword = src[fIndex]
                    new_eword = tgt[eIndex]

                return new_fword, new_eword

    def get_new_tgt(self, fIndex, eIndex, src, tgt):
        if self.replace_both:
            return self.get_new_src_tgt(fIndex, eIndex, src, tgt)

        translations = self.lexf2e.get_translations(src[fIndex], only_in_vocab=True)
        # translations is sorted by lex prob in descending order

        if self.lex_prob_threshold is not None and len(translations) > 0 and translations[0][1] < self.lex_prob_threshold:
            logger.debug("Translations for %s are invalidated because the lexical translation probability(%f) is less than %f" % (src[fIndex], translations[0][1] , self.lex_prob_threshold))
            candidates = []
        else:
            candidates = list(zip_longest([], translations, fillvalue=(src[fIndex], 1.0)))

        logger.debug("Candidates are %s" % (str(candidates),))

        if len(candidates) > 0:
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            if self.tgt_sim_threshold is not None:
                valid = self._is_tgt_replace_valid(tgt[eIndex], best_eword)
                if not valid:
                    logger.debug("Similarity(%f) of e(%s) and e'(%s) is less than %f, so not replacing." % (similarity, tgt[eIndex], best_eword, self.tgt_sim_threshold,))
                    if self.backoff_to_numbered_unk:
                        best_fword = best_eword = self.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        best_fword = src[fIndex]
                        best_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))

            return best_fword, best_eword
        else:
            if self.setting.guarantee_in_vocab_replace:
                logger.debug("Trying to replace both src and tgt...")
                new_fword, new_eword = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                # assert new_fword in self.src_dic and new_eword in self.tgt_dic
                return new_fword, new_eword
            else:
                if self.setting.backoff_to_unk:
                    new_fword = new_eword = self.setting.unk_tag
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                else:
                    logger.debug("No replacement because of no candidates")
                    new_fword = src[fIndex]
                    new_eword = tgt[eIndex]

                return new_fword, new_eword

    def _is_tgt_replace_valid(self, old_eword, new_eword):
        assert self.tgt_sim_threshold is not None
        assert self.tgt_embedding is not None, "You have to set target side word embedding model in order to invalidate the best candidate."
        try:
            similarity = self.tgt_embedding.model.similarity(old_eword, new_eword)
        except KeyError:
            if new_eword not in self.tgt_embedding.vocab:
                logger.info("%s not found in target-size word embedding model" % (new_eword,))
            else:
                logger.info("%s not found in target-size word embedding model" % (old_eword,))
            similarity = 0.0

        if similarity < self.tgt_sim_threshold:
            return False
        else:
            return True


class SimpleReplacer(object):
    def __init__(self, **kwargs):
        self.setting = DotMap(kwargs)
        self.src_voc, self.tgt_voc = self._load_vocab()
        self.src_embedding = self._load_src_embedding()
        self.tgt_embedding = self._load_tgt_embedding()
        self.lexe2f, self.lexf2e = self._load_lex_tables()

        self.lexf2e.set_vocab(self.tgt_voc)
        if self.setting.guarantee_in_vocab_replace:
            filtered_src_voc = self.lexf2e.filter_vocab(self.src_voc)
            logger.info("The size of filtered source vocab: %d" % len(filtered_src_voc))
            self.src_embedding.set_vocab(filtered_src_voc) # set filtered_src_voc to src_embedding also when processing test data
        else:
            self.src_embedding.set_vocab(self.src_voc)

    def _load_src_embedding(self):
        return Word2Vec(model_path=self.setting.src_embedding, topn=self.setting.src_embedding_topn)

    def _should_load_tgt_embedding(self):
        if self.setting.tgt_sim_threshold is not None:
            return True

        return False

    def _load_tgt_embedding(self):
        if self._should_load_tgt_embedding():
            return Word2Vec(model_path=self.setting.tgt_embedding)
        else:
            return None

    def _load_lex_tables(self):
        lexe2f = LexicalDictionary.create_lexical_table(self.setting.dictionary + ".e2f", topn=self.setting.dictionary_topn)
        lexf2e = LexicalDictionary.create_lexical_table(self.setting.dictionary + ".f2e", topn=self.setting.dictionary_topn)
        return lexe2f, lexf2e

    def _load_vocab(self):
        if self.setting.vocab_file is not None and os.path.isfile(self.setting.vocab_file):
            logger.info("Loading vocabularies from %s" % self.setting.vocab_file)
            with open(self.setting.vocab_file, "r") as f:
                vocab_data = json.load(f)
                assert len(vocab_data) == 2
                src_vcb = set(vocab_data[0])
                tgt_vcb = set(vocab_data[1])

            assert self.setting.src_voc_size is None or len(src_vcb) == self.setting.src_voc_size
            assert self.setting.tgt_voc_size is None or len(tgt_vcb) == self.setting.tgt_voc_size

            if self.setting.only_test:
                return src_vcb, None
            else:
                return src_vcb, tgt_vcb

        logger.info("Building vocabularies from training data")
        assert self.setting.src_file_for_voc is not None and self.setting.tgt_file_for_voc is not None, "Both source and target file is needed to build vocabulary"
        src_vcb = build_vocab(self.setting.src_file_for_voc, voc_limit=self.setting.src_voc_size)
        tgt_vcb = build_vocab(self.setting.tgt_file_for_voc, voc_limit=self.setting.tgt_voc_size)

        assert len(src_vcb) == self.setting.src_voc_size
        assert len(tgt_vcb) == self.setting.tgt_voc_size

        if self.setting.vocab_file is not None:
            logger.info("Writing vocab to %s" % self.setting.vocab_file)
            with open(self.setting.vocab_file, "w"):
                json.dump([list(src_vcb), list(tgt_vcb)])

            logger.info("Finished writing vocab to file")

        return src_vcb, tgt_vcb

    def replace(self, src, tgt=None, align=None):
        """
        – unk to unk, both the source and target word in the aligned
          pair are rare words. In this case we will replace the
          source word with a similar in-vocabulary word and the
          target word with the translation of the similar word.
        – unk to common, only the source word is rare. In this
          case we will keep the target word and replace the source
          word with the translation of the target word.
        – common to unk, only the target word is rare. In this case
          we will keep the source word and replace the target word
          with the translation of the source word.
        – common to common, no replacement in this case.
        – unk to null or null to unk, source or target rare word is
          not aligned to any word. In this case we simply remove
          the rare word from the sentence.
        """

        if tgt is not None:
            assert align is not None
            assert isinstance(self.src_voc, set), type(self.src_voc)
            assert isinstance(self.tgt_voc, set), type(self.tgt_voc)

            fwords = list(src)
            ewords = list(tgt)
            one_to_one_links = Utility.get_one_to_one_links(src, tgt, align, self.src_voc, self.tgt_voc)
            for (fIndex, is_src_unk), (eIndex, is_tgt_unk) in one_to_one_links:
                if is_src_unk:
                    if is_tgt_unk:
                        fwords[fIndex], ewords[eIndex] = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                    else:
                        fwords[fIndex], ewords[eIndex] = self.get_new_src(fIndex, eIndex, src, tgt)
                else:
                    if is_tgt_unk:
                        fwords[fIndex], ewords[eIndex] = self.get_new_tgt(fIndex, eIndex, src, tgt)
                    else:
                        fwords[fIndex] = src[fIndex]
                        ewords[eIndex] = tgt[eIndex]

            fwords = filter(None, fwords)
            ewords = filter(None, ewords)
            return fwords, ewords
        else:
            assert self.src_embedding is not None
            assert isinstance(self.src_voc, set)

            fwords = []

            for fIndex, fword in enumerate(src):
                if fword not in self.src_voc:
                    best_fword = self.get_src_substitution(src, fIndex)
                    fwords.append(best_fword)
                else:
                    fwords.append(fword)

            return fwords

    def get_new_src_tgt(self, fIndex, eIndex, src, tgt):
        candidates = []
        most_sim_words = self.src_embedding.most_similar_word(src[fIndex])

        if self.replace_both and src[fIndex] in self.src_dic:
            most_sim_words.insert(0, (src[fIndex], 1.0))

        logger.debug("most similar words of %s in the src vocab are %s" % (src[fIndex], str(most_sim_words)))
        if self.src_sim_threshold is not None and len(most_sim_words) > 0 and most_sim_words[0][1] < self.src_sim_threshold:
            logger.debug("Similarity(%f) to most similar word is less than %f, so not replacing." % (most_sim_words[0][1], self.src_sim_threshold,))
            most_sim_words = []

        for most_sim_word, cos_sim in most_sim_words:
            assert most_sim_word == src[fIndex] or most_sim_word in self.src_dic, "%s is not in the dictionary!" % most_sim_word
            translations = self.lexf2e.get_translations(most_sim_word, only_in_vocab=True)
            if len(translations) > 0:
                candidates += list(zip_longest([(most_sim_word, cos_sim)], translations, fillvalue=(most_sim_word, cos_sim)))

        logger.debug("Candidates are %s" % (str(candidates),))

        if len(candidates) > 0:
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            new_fword = best_fword
            new_eword = best_eword
            if self.tgt_sim_threshold is not None:
                valid = self._is_tgt_replace_valid(tgt[eIndex], new_eword)
                if not valid:
                    logger.debug("Similarity of e(%s) and e'(%s) is less than %f, so not replacing." % (tgt[eIndex], new_eword, self.tgt_sim_threshold,))
                    if self.setting.backoff_to_unk:
                        new_fword = new_eword = self.setting.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        new_fword = src[fIndex]
                        new_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))

            assert (src[fIndex] == best_fword or (best_fword in self.src_dic)), "%s->%s" % (src[fIndex], best_fword)
        else:
            if self.setting.backoff_to_unk:
                new_fword = new_eword = self.setting.unk_tag
                logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
            else:
                logger.debug("No replacement because of no candidates")
                new_fword = src[fIndex]
                new_eword = tgt[eIndex]

        return new_fword, new_eword

    def get_new_src(self, fIndex, eIndex, src, tgt):
        if self.replace_both:
            return self.get_new_src_tgt(fIndex, eIndex, src, tgt)

        translations = self.lexe2f.get_translations(tgt[eIndex], only_in_vocab=True)

        if self.lex_prob_threshold is not None and len(translations) > 0 and translations[0][1] < self.lex_prob_threshold:
            logger.debug("Translations for %s are invalidated because the lexical translation probability(%f) is less than %f" % (tgt[eIndex], translations[0][1], self.lex_prob_threshold))
            translations = []

        if len(translations) > 0:
            probs = [translation[1] for translation in translations]
            tgt_candidates = list(zip_longest([tgt[eIndex]], probs ,fillvalue=tgt[eIndex]))
            for i in range(len(translations)):
                translations[i] = (translations[i][0], self.src_embedding.similarity(translations[i][0], src[fIndex]))
            candidates = list(zip_longest(translations, tgt_candidates))
            logger.debug("Candidates are %s" % (str(candidates),))
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            if self.src_sim_threshold is not None:
                try:
                    similarity = self.src_embedding.similarity(src[fIndex], best_fword)
                except KeyError:
                    logger.debug("Either %s or %s are not in the source vocab" % (src[fIndex], best_fword))
                    similarity = 0.0

                if similarity < self.setting.src_sim_threshold:
                    logger.debug("Similarity(%f) between %s and %s is less than %f, so not replacing." % (similarity, src[fIndex], best_fword, self.src_sim_threshold,))
                    if self.setting.backoff_to_unk:
                        best_fword = best_eword = self.setting.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        best_fword = src[fIndex]
                        best_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))
            else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))
            return best_fword, best_eword
        else:
            if self.setting.guarantee_in_vocab_replace:
                logger.debug("Trying to replace both src and tgt...")
                new_fword, new_eword = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                # assert new_fword in self.src_dic and new_eword in self.tgt_dic
                return new_fword, new_eword
            else:
                if self.backoff_to_unk:
                    new_fword = new_eword = self.unk_tag
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                else:
                    logger.debug("No replacement because of no candidates")
                    new_fword = src[fIndex]
                    new_eword = tgt[eIndex]

                return new_fword, new_eword

    def get_new_tgt(self, fIndex, eIndex, src, tgt):
        if self.replace_both:
            return self.get_new_src_tgt(fIndex, eIndex, src, tgt)

        translations = self.lexf2e.get_translations(src[fIndex], only_in_vocab=True)
        # translations is sorted by lex prob in descending order

        if self.lex_prob_threshold is not None and len(translations) > 0 and translations[0][1] < self.lex_prob_threshold:
            logger.debug("Translations for %s are invalidated because the lexical translation probability(%f) is less than %f" % (src[fIndex], translations[0][1] , self.lex_prob_threshold))
            candidates = []
        else:
            candidates = list(zip_longest([], translations, fillvalue=(src[fIndex], 1.0)))

        logger.debug("Candidates are %s" % (str(candidates),))

        if len(candidates) > 0:
            best_fword, best_eword = self.get_best_src_tgt_pair(src, tgt, fIndex, eIndex, candidates)
            if self.tgt_sim_threshold is not None:
                valid = self._is_tgt_replace_valid(tgt[eIndex], best_eword)
                if not valid:
                    logger.debug("Similarity(%f) of e(%s) and e'(%s) is less than %f, so not replacing." % (similarity, tgt[eIndex], best_eword, self.tgt_sim_threshold,))
                    if self.backoff_to_numbered_unk:
                        best_fword = best_eword = self.unk_tag
                        logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                    else:
                        logger.debug("No replacement because of no candidates")
                        best_fword = src[fIndex]
                        best_eword = tgt[eIndex]
                else:
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], best_fword, best_eword))

            return best_fword, best_eword
        else:
            if self.setting.guarantee_in_vocab_replace:
                logger.debug("Trying to replace both src and tgt...")
                new_fword, new_eword = self.get_new_src_tgt(fIndex, eIndex, src, tgt)
                # assert new_fword in self.src_dic and new_eword in self.tgt_dic
                return new_fword, new_eword
            else:
                if self.setting.backoff_to_unk:
                    new_fword = new_eword = self.setting.unk_tag
                    logger.debug("[Replacement] (%s, %s) → (%s, %s)" % (src[fIndex], tgt[eIndex], new_fword, new_eword))
                else:
                    logger.debug("No replacement because of no candidates")
                    new_fword = src[fIndex]
                    new_eword = tgt[eIndex]

                return new_fword, new_eword

    def _is_tgt_replace_valid(self, old_eword, new_eword):
        assert self.tgt_sim_threshold is not None
        assert self.tgt_embedding is not None, "You have to set target side word embedding model in order to invalidate the best candidate."
        try:
            similarity = self.tgt_embedding.model.similarity(old_eword, new_eword)
        except KeyError:
            if new_eword not in self.tgt_embedding.vocab:
                logger.info("%s not found in target-size word embedding model" % (new_eword,))
            else:
                logger.info("%s not found in target-size word embedding model" % (old_eword,))
            similarity = 0.0

        if similarity < self.tgt_sim_threshold:
            return False
        else:
            return True
