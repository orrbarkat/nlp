from abc import ABC, abstractmethod


class Summarizer(ABC):
    @abstractmethod
    def summarize(self, text, ratio=0.2, word_count=None, split=False):
        pass
