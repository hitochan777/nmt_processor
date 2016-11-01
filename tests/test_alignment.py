from unittest import TestCase

from alignment import Alignment


class TestAlignment(TestCase):
    def test_convert_string_to_alignment_dictionary1(self):
        line = "1-2 2-1 3-2 4-1 1-3"
        alignment = Alignment.convert_string_to_alignment_dictionary(line)
        self.assertDictEqual(alignment, {1: [2, 3], 2: [1], 3: [2], 4: [1]})

    def test_convert_string_to_alignment_dictionary2(self):
        # Empty line
        line = ""
        alignment = Alignment.convert_string_to_alignment_dictionary(line)
        self.assertDictEqual(alignment, {})
