from project.summarizers.fast_abs_rl import FastAbsRl
from project.summarizers.textrank import TextRank


def summarizer_factory(model_class):
    classes = {
        'textrank': TextRank,
        'fast_rl': FastAbsRl
    }
    summarizer = classes[model_class.lower()]()
    return summarizer
