from aspects.utilities.nlp import load_spacy

nlp = load_spacy()


def preprocess(text):
    # remove special chars from RST parser
    text = text[2:-2]
    doc = nlp(unicode(text))
    return doc
