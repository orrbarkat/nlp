from sumeval.metrics.rouge import RougeCalculator
import sys, json
import argparse


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def calc_rouge(machine_summery, reference_summery, debug_print = False):
    rouge = RougeCalculator(stopwords=True, lang="en")

    rouge_1 = rouge.rouge_n(
        summary=machine_summery,
        references=reference_summery,
        n=1)

    rouge_2 = rouge.rouge_n(
        summary=machine_summery,
        references=reference_summery,
        n=2)

    rouge_l = rouge.rouge_l(
        summary=machine_summery,
        references=reference_summery)

    if debug_print:
        print("current sentences results:\nROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
            rouge_1, rouge_2, rouge_l
        ).replace(", ", "\n"))

    return rouge_1, rouge_2, rouge_l


def print_sentences(j_ref_line, j_sum_line):
    print("")
    print("machine made: ")
    print(j_sum_line)
    print("")
    print("reference summery was:")
    print(j_ref_line)
    print("")


def apply_rouge(sum_line, verbose):

    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    count = 0

    for i, entry in enumerate(sum_line):
        entry = json.loads(entry)

        j_sum_line = entry["sum"]
        j_ref_line = entry["ref"]

        if j_sum_line is None:
            print(bcolors.WARNING + "PROBLEM: empty summary in line #" + str(i + 1) + ")\n" + bcolors.ENDC)
            continue

        if j_ref_line is None:
            print(bcolors.WARNING + "PROBLEM: empty reference summary in line #" + str(i + 1) + ")\n" + bcolors.ENDC)
            continue

        if verbose:  # print sentences for extra information
            print("...\n\non iteration " + str(i+1) + ":")
            print_sentences(j_ref_line, j_sum_line)

        results = calc_rouge(j_sum_line, j_ref_line, verbose)

        rouge_1 += results[0]
        rouge_2 += results[1]
        rouge_l += results[2]
        count += 1

    return "ROUGE-1: {},  ROUGE-2: {},  ROUGE-L: {}".format(
        rouge_1 / count, rouge_2 / count, rouge_l / count  # average
    ).replace(", ", "\n")


def eval(args):  # main evaluation function
    sum_line = args.input.readlines()
    verbose = args.verbose

    # Rouge
    if args.model == "rouge":
        result_msg = apply_rouge(sum_line, verbose)

    # Else
    print(bcolors.OKGREEN + "\nEvaluation method:", str(args.model), "\n\nResults are:\n", result_msg + bcolors.ENDC, file=args.output, flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='evaluate summarise')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('eval', help='', )
    command_parser.add_argument('-d', '--data', type=str, default="data/dev.data",
                               help="Evaluation data")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout,
                                help="evaluation output file")
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default="summarise.txt",
                                help="summarise input file (output of the summary part)")
    command_parser.add_argument('-m', '--model', choices=["rouge"], default="rouge",
                                help="module that should evaluate the resulting summaries")
    command_parser.add_argument('-v', '--verbose', action='store_true', default=False,
                                help='Display prediction probabilities')

    #command_parser.add_argument('-n', '--num-texts', type=int,
    #                            help="Evaluation data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")

    command_parser.set_defaults(func=eval)

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

