from sumeval.metrics.rouge import RougeCalculator
import sys, json
from newsroom import jsonl

def calc_rouge(machine_summery, reference_summery, debugPrint = False):
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

    if debugPrint:
        print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
            rouge_1, rouge_2, rouge_l
        ).replace(", ", "\n"))

    return tuple(rouge_1, rouge_2, rouge_l)

def calc_rouge_vec(machine_summery, reference_summery, debugPrint = False):
    rouge = RougeCalculator(stopwords=True, lang="en")
    ans = []
    for summery, reference in zip(machine_summery, reference_summery):
        rouge_1 = rouge.rouge_n(
            summary=summery,
            references=reference,
            n=1)

        rouge_2 = rouge.rouge_n(
            summary=summery,
            references=reference,
            n=2)

        rouge_l = rouge.rouge_l(
            summary=summery,
            references=reference)

        if debugPrint:
            print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
                rouge_1, rouge_2, rouge_l
            ).replace(", ", "\n"))

        ans.append(tuple(rouge_1, rouge_2, rouge_l))
    return ans


def eval(args):  # main evaluation function

    #sum_file_name = str(sys.argv[1])

    #ref_file_name = str(sys.argv[2])

    #if len(sys.argv) > 3:
    #    output_file_name = sys.argv[3]

    sum_lines_file = args.input
    ref_file_name = args.data

    ##
    with jsonl.open(ref_file_name, gzip=True) as original_dataset_file:
        sum_line = sum_lines_file.readlines()
        for index, entry in enumerate(original_dataset_file):
            j_ref_line = entry["summary"]
            j_sum_line = json.loads(sum_line[index])

            if args.verbose == True:  # print sentences for extra information
                print("")
                print("machine made: ")
                print(j_sum_line)
                print("")
                print("reference summery was:")
                print(j_ref_line)
                print("")
                print("...")



if __name__ == '__main__':

    ## new
    import argparse

    parser = argparse.ArgumentParser(description='evaluate summarise')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('eval', help='', )
    command_parser.add_argument('-d', '--data', type=str, default="data/dev.data",
                               help="Evaluation data")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout,
                                help="evaluation output file")
    command_parser.add_argument('-i', '--input', type=argparse.FileType('r'), default="summarise.txt",
                                help="summarise input file (output of the summary part)")
    command_parser.add_argument('-m', '--model-class', choices=["rouge"], default="rouge",
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

