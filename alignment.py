import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Alignment(object):

    @classmethod
    def convert_string_to_alignment_dictionary(cls, line):
        dic = defaultdict(list)
        if line.strip() == "":
            return {}

        links = line.rstrip().split(" ")
        for link in links:
            fword, eword = list(map(int, link.split("-")))
            dic[fword].append(eword)

        return dic
        
    @classmethod
    def read_alignment(cls, filename):
        if isinstance(filename, str):
            f = open(filename, "r")
        else:
            f = filename

        for line in f:
            dic = cls.convert_string_to_alignment_dictionary(line)
            yield dic
