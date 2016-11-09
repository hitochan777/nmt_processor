import argparse

from replacer import Replacer, ReplaceMethod


def command_line(arguments=None):
    parser = argparse.ArgumentParser("Replacer")
    parser.add_argument("--mode", required=True, type=str, choices=[val.name for val in ReplaceMethod])
    parser.add_argument("--save-prefix", required=True, type=str, help="prefix of path to save files")
    parser.add_argument("--src-embedding", type=str, default=None, help="path to word2vec model for source language")
    parser.add_argument("--src-embedding-topn", default=10, type=int, help="Use N most similar words as candidates")
    parser.add_argument("--tgt-embedding", type=str, default=None, help="path to word2vec model for target language")
    parser.add_argument("--src-voc-size", type=int, default=None, help="source vocabulary size")
    parser.add_argument("--tgt-voc-size", type=int, default=None, help="target vocabulary size")
    parser.add_argument("--src-train-file", type=str, default=None, help="path to source training file")
    parser.add_argument("--tgt-train-file", type=str, default=None, help="path to target training file")
    parser.add_argument("--src-dev-file", type=str, default=None, help="path to source dev file")
    parser.add_argument("--tgt-dev-file", type=str, default=None, help="path to target dev file")
    parser.add_argument("--src-test-file", type=str, default=None, help="path to source test file")
    # parser.add_argument("--tgt-test-file", type=str, default=None, help="path to target test file")
    parser.add_argument("--vocab-file", type=str, default=None, help="Path to vocabulary file (JSON style)")
    parser.add_argument("--subset-train", type=int, default=None, help="If this option is specified, the first N lines of training data will be processed")
    parser.add_argument("--align-train", type=str, default=None, help="Path to word alignment file for training data")
    parser.add_argument("--align-dev", type=str, default=None, help="Path to word alignment file for dev data")
    parser.add_argument("--dictionary", type=str, default=None, help="Path to dictionary file (JSON style)")
    parser.add_argument("--dictionary-topn", default=None, type=int, help="Use N translations with highest probability")
    parser.add_argument("--src-sim-threshold", type=float, default=None, help="threshold value for source side cosine similarity")
    parser.add_argument("--tgt-sim-threshold", type=float, default=None, help="threshold value for target side cosine similarity")
    parser.add_argument("--lex-prob-threshold", type=float, defalt=None, help="threshold value for lexical probability")
    parser.add_argument("--backoff-to-unk", action='store_true', help="")
    parser.add_argument("--unk-tag", type=str, default="@UNK", "UNK tag to use when backoff-to-unk is True")
    parser.add_argument("--replace-both", action='store_true', help="Force replace both linked source and target word even if only one of them is unknown")
    parser.add_argument("--guarantee-in-vocab-replace", action='store_true', help="Guarantee that replacing words are all in the vocabulary")

    args = parser.parse_args(arguments)

    return args


def replace_pair(replacer, src, tgt, align):

    pass


def replace_single(replacer, src):
    pass


def main():
    # http://stackoverflow.com/questions/15206010/how-to-pass-on-argparse-argument-to-function-as-kwargs
    args = command_line()
    assert (args.src_train_file is None) == (args.tgt_train_file is None)
    assert (args.src_dev_file is None) == (args.tgt_dev_file is None)
    # above 2 assertion lines are needed to guarantee that src_train_file being None implies tgt_train_file being None and so on...
    only_test = (args.src_train_file is None) and (args.src_dev_file is None)
    replacer = Replacer(args, mode=args.mode, src_embedding=args.src_embedding,
                        src_embedding_topn=args.src_embedding_topn, tgt_embedding=args.tgt_embedding,
                        src_voc_size=args.src_voc_size, tgt_voc_size=args.tgt_voc_size, only_test=only_test,
                        dictionary=args.dictionary, dictionary_topn=args.dictionary_topn,
                        src_sim_threshold=args.src_sim_threshold, tgt_sim_threshold=args.tgt_sim_threshold,
                        lex_prob_threshold=args.lex_prob_threshold, backoff_to_unk=args.backoff_to_unk,
                        replace_both=args.replace_both, guarantee_in_vocab_replace=args.guarantee_in_vocab_replace)

if __name__ == "__main__":
    main()
