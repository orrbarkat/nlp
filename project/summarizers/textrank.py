from gensim.summarization.summarizer import summarize as textrank

from project.summarizers.base import Summarizer


class TextRank(Summarizer):
    WORD_COUNT = 50

    def summarize(self, text, ratio=0.2, word_count=WORD_COUNT, split=False):
        return textrank(text, word_count=self.WORD_COUNT)
