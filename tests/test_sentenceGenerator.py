from unittest import TestCase
from io import StringIO

from word_embedding import SentenceGenerator


class TestSentenceGenerator(TestCase):
    def test_sentence_generator(self):
        raw_text = (
            "this is a pen .\n"
            "it is raining today .\n"
            "What a day !"
        )
        fs = StringIO(raw_text)
        gen = SentenceGenerator(fs)
        for index, line in enumerate(gen):
            self.assertEqual(line, raw_text.split("\n")[index].split(" "))


