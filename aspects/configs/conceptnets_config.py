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
CONCEPTNET_API_URL = u'http://api.conceptnet.io'
CONCEPTNET_URL = u'{}/c/{}/'.format(CONCEPTNET_API_URL, CONCEPTNET_LANG)
CONCEPTNET_RELATIONS = [u'LocatedNear', u'HasA', u'PartOf', u'MadeOf', u'IsA',
                        u'InheritsFrom', u'Synonym']
