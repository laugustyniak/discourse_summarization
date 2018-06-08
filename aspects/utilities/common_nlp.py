import spacy


def load_spacy(model='en_core_web_sm'):
    return spacy.load(model)


ASPECTS_TO_SKIP = [
    u'day', u'days', u'week', u'weeks',
    u'tonight',
    u'total', u'laughter', u'tongue',
    u'weekend', u'month', u'months', u'year',
    u'years', u'time', u'today', u'data',
    u'date',
    u'monday', u'tuesday', u'wednesday',
    u'thursday', u'friday', u'saturday',
    u'sunday',
    u'january', u'february', u'march', u'april',
    u'may', u'june', u'july', u'august',
    u'september', u'october',
    u'november', u'december',
    u'end',
    u'', u't',
    u'noise',
    u'customer', u'agent',
    u'unk',
    u'password',
    u'don',
]
