import collections
import operator
import logging
import io

logger = logging.getLogger(__name__)


def build_vocab(fs, voc_limit=None, max_nb_ex=None):
    if isinstance(fs, str):
        f = open(fs, "r")
    else:
        f = fs

    if max_nb_ex is not None:
        logger.info("Using the first %d lines in training data" % max_nb_ex)

    counts = collections.defaultdict(int)
    for num_ex, line in enumerate(f):
        if max_nb_ex is not None and num_ex >= max_nb_ex:
            break
        line = line.rstrip().split(" ")
        for w in line:
            counts[w] += 1

    sorted_counts = sorted(
        counts.items(), key=operator.itemgetter(1, 0), reverse=True)

    voc = set(map(lambda x: x[0], sorted_counts[:voc_limit]))
    return voc
