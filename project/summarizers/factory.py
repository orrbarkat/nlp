from project.summarizers.textrank import TextRank


def summarizer_factory(model_class):
    classes = {
        'textrank': TextRank
    }
    summarizer = classes[model_class.lower()]()
    return summarizer
