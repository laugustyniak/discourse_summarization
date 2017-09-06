# sentic net conceptnet
# Do we use sentic conceptnet based entities in aspect extraction procedure.
# We use sentic.net as data source here.
SENTIC_ASPECTS = True
SENTIC_EXACT_MATCH_CONCEPTS = True

# conceptnet io
# Do we use ConceptNet based entities in aspect extraction procedure.
# We use conceptnet.io as data source here.
CONCEPTNET_ASPECTS = True
CONCEPTNET_LANG = u'en'
CONCEPTNET_EXACT_MATCH_CONCEPTS = True
CONCEPTNET_URL = u'http://api.conceptnet.io/c/{}/'.format(CONCEPTNET_LANG)
CONCEPTNET_RELATIONS = ['LocatedNear', 'HasA', 'PartOf', 'MadeOf', 'IsA',
                        'InheritsFrom', 'Synonym']
