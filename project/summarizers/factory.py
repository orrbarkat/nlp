from summarizers.fast_abs_rl import FastAbsRl
from summarizers.textrank import TextRank
from summarizers.lede3 import Lede3




def summarizer_factory(model_class):
    classes = {
        'textrank': TextRank,
        'fast_rl': FastAbsRl,
        'lede3': Lede3
    }
    summarizer = classes[model_class.lower()]()
    return summarizer
