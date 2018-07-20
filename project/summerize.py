import json
import sys

from gensim.summarization.summarizer import summarize
from newsroom import jsonl


def evaluate(args):
    WORD_COUNT = 50

    with jsonl.open(args.data, gzip=True) as dataset:
        for i, entry in enumerate(dataset):
            if args.num_texts is not None and i >= args.num_texts:
                break

            try:
                summary = summarize(entry["text"], word_count=WORD_COUNT)

            except ValueError:
                # Handles "input must have more than one sentence"
                summary = entry["text"]

            print(json.dumps(summary), file=args.output, flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Summarize files or console')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('evaluate', help='', )
    command_parser.add_argument('-d', '--data', type=str, default="data/dev.data",
                                help="Evaluation data")
    command_parser.add_argument('-n', '--num-texts', type=int, help="process only n records")
    command_parser.add_argument('-m', '--model-class', help="module that should summarize")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.add_argument('-v', '--verbose', action='store_true', default=False,
                                help='Display prediction probabilities')
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=evaluate)

    # command_parser = subparsers.add_parser('shell', help='')
    # command_parser.add_argument('-m', '--model-path', help="Training data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    # command_parser.add_argument('-s', '--verbose', action='store_true', default=False, help='Display prediction probabilities')
    # command_parser.set_defaults(func=do_shell)

    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        args.func(args)
