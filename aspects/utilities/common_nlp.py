try:
    from repoze.lru import lru_cache
except:
    from functools import lru_cache
import RAKE
import spacy


@lru_cache(None)
def load_spacy(model='en_core_web_sm'):
    return spacy.load(model)


ASPECTS_TO_SKIP = [
    u'day', u'days', u'week', u'weeks',
    u'tonight', u'tomorrow',
    u'total', u'laughter', u'tongue',
    u'weekend', u'month', u'months', u'year', u'years', u'time', u'today', u'data', u'date',
    u'monday', u'tuesday', u'wednesday', u'thursday', u'friday', u'saturday', u'sunday',
    u'january', u'february', u'march', u'april', u'may', u'june', u'july', u'august',
    u'september', u'october', u'november', u'december',
    u'start', u'end',
    u't',
    u'noise',
    u'customer', u'agent',
    u'unk',
    u'password',
    u'don',
    u'other', u'others'
]


@lru_cache(None)
def load_rake():
    return RAKE.Rake(RAKE.SmartStopList())


@lru_cache(2048)
def spelling(word):
    if word in SPELLING_CORRECTION:
        return SPELLING_CORRECTION[word]
    else:
        return word


SPELLING_CORRECTION = {
    'anteanna': 'antenna',
    'arm band:': 'armband',
    'auto load': 'autoload',
    'auto-focus': 'autofocus',
    'bargins': 'bargain',
    'bluetooth': 'Bluetooth',
    'documentatio': 'documentation',
    'e-mail': 'email',
    'hand set': 'handset',
    'i-pod': 'iPod',
    'i-tunes': 'iTunes',
    'ipod': 'iPod',
    'itunes': 'iTunes',
    'joy stick': 'joystick',
    'labling': 'sabling',
    'm12': '',
    'mmc': 'MMC',
    'navigatee': 'navigates',
    'nokia': 'Nokia',
    'norton': 'Norton',
    'nstalling': 'installation',
    'pail.': 'pail',
    'plaeyr': 'player',
    'powerup': 'power',
    'responce': 'response',
    'semelled': 'smelled',
    'sensitivy': 'sensitive',
    'set up': 'setup',
    'set-up': 'setup',
    'smal': 'small',
    'sony': 'Sony',
    'speaker phone': 'speakerphone',
    'srorage': 'storage',
    'start up': 'startup',
    'start-up': 'startup',
    'storage': 'storage',
    'tft': 'TFT',
    'touch pad': 'touchpad',
    'touch screen': 'touchscreen',
    'touch-pad': 'touchpad',
    'trnasfer': 'transfer',
    'unstalling': 'uninstallation',
    'upgradibility': 'upgradability',
    'usb': 'USB',
    'verizon': 'Verizon',
    'web site': 'website',
}
