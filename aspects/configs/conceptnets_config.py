# sentic net conceptnet
# Do we use sentic conceptnet based entities in aspect extraction procedure.
# We use sentic.net as data source here.
SENTIC_ASPECTS = True
SENTIC_EXACT_MATCH_CONCEPTS = True

# conceptnet io
# Do we use ConceptNet based entities in aspect extraction procedure.
# We use conceptnet.io as data source here.
CONCEPTNET_ASPECTS = True
CONCEPTNET_EXACT_MATCH_CONCEPTS = True
CONCEPTNET_URL = 'http://api.conceptnet.io/c/en/'
CONCEPTNET_RELATIONS = ['LocatedNear', 'HasA', 'PartOf', 'MadeOf', 'IsA',
                        'InheritsFrom']
