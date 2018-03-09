import logging
from collections import defaultdict

import RAKE
import requests
from simplejson import JSONDecodeError

import aspects.configs.conceptnets_config as config
from aspects.enrichments.conceptnets import Sentic, ConceptNetIO

log = logging.getLogger(__name__)

ASPECTS_TO_SKIP = [u'day', u'days', u'week', u'weeks',
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

rake = RAKE.Rake(RAKE.SmartStopList())
cn = ConceptNetIO()
cn.load_cnio()


def extract_noun_and_noun_phrases(df, column_with_text='edu'):
    # FIXME: reimplement it
    all_edus_aspects = []
    for words in df[column_with_text]:
        aspects = []
        aspect_sequence = []
        aspect_sequence_main_encountered = False
        aspect_sequence_enabled = False
        for token in words:
            if token.pos_ == 'NOUN' and len(token) > 1:
                if not token.is_stop:
                    aspect_sequence.append(token.lemma_)
                aspect_sequence_enabled = True
                aspect_sequence_main_encountered = True
            else:
                if aspect_sequence_enabled and aspect_sequence_main_encountered:
                    aspect = ' '.join(aspect_sequence)
                    if aspect not in aspects:
                        aspects.append(aspect)
                aspect_sequence_main_encountered = False
                aspect_sequence_enabled = False
                aspect_sequence = []
        if aspect_sequence_enabled and aspect_sequence_main_encountered:
            aspects.append(' '.join(aspect_sequence))
        # filter empty strings
        aspects = [aspect.lower() for aspect in aspects if aspect]
        all_edus_aspects.append(aspects)
    df['aspects'] = all_edus_aspects
    return df


def extract_named_entities(df, column_with_text='edu'):
    # edu consists spacy objects for whole document
    df['named_entities'] = [edu.ents if edu.ents else None for edu in df[column_with_text]]
    return df


def extract_sentic_concepts(df):
    sentic = Sentic()
    sentic_concepts = []
    for aspects in df['aspects']:
        concept_aspects = {}
        for aspect in aspects:
            aspect = aspect.replace(' ', '_')
            concept_aspects.update(sentic.get_semantic_concept_by_concept(aspect, config.SENTIC_EXACT_MATCH_CONCEPTS))
        sentic_concepts.append(concept_aspects)
    df['sentic_aspects'] = sentic_concepts
    return df


def extract_keywords_rake(df):
    df['rake_keywords'] = [rake.run(edu.text) if edu.text else None for edu in df.edu]
    return df


def extract_conceptnet_concepts(df):
    conceptnet_concepts = []
    for n_doc, aspects in enumerate(df.aspects):
        concept_aspects = defaultdict(list)
        for aspect in aspects:
            if aspect not in cn.concepts_io:
                concept_aspects[aspect] = []
                next_page = config.CONCEPTNET_URL + aspect + config.CONCEPTNET_API_URL_OFFSET_AND_LIMIT
                n_pages = 1
                while next_page:
                    next_page = next_page.replace(' ', '_')
                    log.info('#{} pages for {}'.format(n_pages, aspect))
                    n_pages += 1
                    try:
                        response = requests.get(next_page).json()
                    except JSONDecodeError as err:
                        log.error('Response parsing error: {}'.format(str(err)))
                        raise JSONDecodeError(str(err))
                    try:
                        cn_edges = response['edges']
                        cn_view = response['view']
                        next_page = config.CONCEPTNET_API_URL + cn_view['nextPage']
                        log.info('Next page from ConceptNet.io: {}'.format(next_page))
                        for edge in cn_edges:
                            relation = edge['rel']['label']
                            if relation in config.CONCEPTNET_RELATIONS \
                                    and (edge['start']['language'] == config.CONCEPTNET_LANG
                                         and edge['end']['language'] == config.CONCEPTNET_LANG):
                                concept_aspects[aspect].append({'start': edge['start']['label'].lower(),
                                                                'start-lang': edge['start']['language'],
                                                                'end': edge['end']['label'].lower(),
                                                                'end-lang': edge['end']['language'],
                                                                'relation': relation,
                                                                'weight': edge['weight']})
                    except KeyError:
                        log.error('Next page url: {} will be set to None'.format(next_page))
                        if 'error' in response.keys():
                            log.error(response['error']['details'])
                        next_page = None
                cn.concepts_io.update(concept_aspects)
                if not n_doc % 100:
                    cn.save_cnio()
            else:
                log.debug('We have already stored this concept: {}'.format(aspect))
                concept_aspects[aspect] = cn.concepts_io[aspect]
        conceptnet_concepts.append(concept_aspects)
    df['conceptnet_concepts'] = [concepts if concepts else None for concepts in conceptnet_concepts]
    return df
