import argparse


def command_line():
    parser = argparse.ArgumentParser("pre- and post- process files")
    parser.add_argument("--mode", required=True, type=str, choices=["simple", "bilm", "bisim", "copy"])
    parser.add_argument("--src-embedding", type=str, help="path to word2vec model for source language")
    parser.add_argument("--src-embedding-topn", default=10, type=int, help="Use N most similar words as candidates")
    parser.add_argument("--tgt-embedding", type=str, help="path to word2vec model for target language")
    parser.add_argument("--src-voc-size", type=int, help="source vocabulary size")
    parser.add_argument("--tgt-voc-size", type=int, help="target vocabulary size")
    parser.add_argument("--src-train-file", type=str, help="path to source training file")
    parser.add_argument("--tgt-train-file", type=str, help="path to target training file")
    parser.add_argument("--src-dev-file", type=str, help="path to source dev file")
    parser.add_argument("--tgt-dev-file", type=str, help="path to target dev file")
    parser.add_argument("--src-test-file", type=str, help="path to source test file")
    parser.add_argument("--tgt-test-file", type=str, help="path to target test file")
    parser.add_argument("--save-prefix", required=True, type=str, help="prefix of path to save files")
    parser.add_argument("--vocab-file", type=str, help="Path to vocabulary file (JSON style)")
    parser.add_argument("--subset-train", type=int, help="If this option is specified, the first N lines of training data will be processed")
    parser.add_argument("--align-train", type=str, help="Path to word alignment file for training data")
    parser.add_argument("--align-dev", type=str, help="Path to word alignment file for dev data")
    parser.add_argument("--dictionary", type=str, help="Path to dictionary file (JSON style)")
    parser.add_argument("--dictionary-topn", default=None, type=int, help="Use N translations with highest probability")
    parser.add_argument("--src-sim-threshold", type=float, help="threshold value for source side cosine similarity")
    parser.add_argument("--tgt-sim-threshold", type=float, help="threshold value for target side cosine similarity")
    parser.add_argument("--lex-prob-threshold", type=float, help="threshold value for lexical probability")
    parser.add_argument("--backoff-to-unk", action='store_true', help="")
    parser.add_argument("--replace-both", action='store_true', help="Force replace both linked source and target word even if only one of them is unknown")

    args = parser.parse_args()
    return args


def main():
    # http://stackoverflow.com/questions/15206010/how-to-pass-on-argparse-argument-to-function-as-kwargs
    args = command_line()
    mode = args.mode
    if mode == "simple":
        pass
    elif mode == "bilm":
        pass
    elif mode == "bisim":
        pass
    elif mode == "copy":
        pass
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
