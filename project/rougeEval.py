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

if __name__ == '__main__':

    sum_file_name = str(sys.argv[1])

    ref_file_name = str(sys.argv[2])

    if len(sys.argv) > 3:
        output_file_name = sys.argv[3]

    with open(sum_file_name) as sum_lines_file, jsonl.open(ref_file_name, gzip=True) as dataset:
        sum_line = sum_lines_file.readlines()
        for i, entry in enumerate(dataset):
            j_ref_line = entry["summary"]
            j_sum_line = json.loads(sum_line[i])

            print("<>")
            print(j_sum_line)
            print("<  >")
            print(j_ref_line)
            print("<      >")
            print(" ")
