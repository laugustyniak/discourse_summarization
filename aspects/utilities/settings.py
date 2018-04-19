import pathlib

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
DATA_PATH = ROOT_PATH / 'data'
EDU_DEPENDENCY_PARSER_PATH = ROOT_PATH.parent / 'edu_dependency_parser'
DEFAULT_OUTPUT_PATH = ROOT_PATH / 'results'
DEFAULT_INPUT_FILE_PATH = ROOT_PATH / 'texts' / 'test.txt'

# AMAZON DATASETS PATHS
AMAZON_REVIEWS_DATASETS_PATH = DATA_PATH / 'reviews' / 'amazon'
AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_GZ = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Apps_for_Android.json.gz'
AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Apps_for_Android.json'
AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_GZ = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Amazon_Instant_Video.json.gz'
AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Amazon_Instant_Video.json'
AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_GZ = \
    AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Cell_Phones_and_Accessories.json.gz'
AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON = \
    AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Cell_Phones_and_Accessories.json'

# SENITMENT MODELS
SENTIMENT_MODELS_PATH = DATA_PATH / 'models'
SENTIMENT_MODEL_PROD = SENTIMENT_MODELS_PATH / \
                       'Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-10-domains.pkl'
# smaller model as default - useful for testing
SENTIMENT_MODEL_TESTS = SENTIMENT_MODELS_PATH / 'Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5' \
                                                '-reviews_Apps_for_Android-500000-balanced.pkl'

# conceptnets
CONCEPTNET_IO_PATH_GZ = DATA_PATH / 'conceptnet' / 'conceptnet-assertions-5.5.5.csv.gz'
CONCEPTNET_IO_PKL = DATA_PATH / 'conceptnet' / 'conceptnet_io.pkl'

# sentic net conceptnet
# Do we use sentic conceptnet based entities in aspect extraction procedure.
# We use sentic.net as data source here.
SENTIC_ASPECTS = True
SENTIC_EXACT_MATCH_CONCEPTS = True

# conceptnet io
# Do we use ConceptNet based entities in aspect extraction procedure.
# We use conceptnet.io as data source here.
CONCEPTNET_IO_ASPECTS = True
CONCEPTNET_IO_LANG = u'en'
CONCEPTNET_IO_RELATIONS = [u'LocatedNear', u'HasA', u'PartOf', u'MadeOf', u'IsA', u'InheritsFrom', u'Synonym']

NER_TYPES = [u'PERSON', u'GPE', u'ORG', u'PRODUCT', u'FAC', u'LOC']

# serialization steps
ASPECT_EXTRACTION_SERIALIZATION_STEP = 10000

# sample trees
SAMPLE_TREE_177 = DATA_PATH / 'sample_trees' / '177'
SAMPLE_TREE_189 = DATA_PATH / 'sample_trees' / '189'
