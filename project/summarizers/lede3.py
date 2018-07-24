from summarizers.base import Summarizer
import re

class Lede3(Summarizer):
    WORD_COUNT = 50

    def summarize(self, text, ratio=0.2, word_count=WORD_COUNT, split=False):
        return ' '.join(re.split(r'(?<=[.:;?!])\s', text)[:3])
