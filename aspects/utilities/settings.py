import pathlib

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent
DATA_PATH = ROOT_PATH / "data"
EDU_DEPENDENCY_PARSER_PATH = ROOT_PATH.parent / "edu_dependency_parser"
DEFAULT_OUTPUT_PATH = ROOT_PATH / "results"
DEFAULT_INPUT_FILE_PATH = ROOT_PATH / pathlib.Path(
    "data/reviews/amazon/reviews_Apps_for_Android.json"
)

# --------------------------------------------- AMAZON DATASETS PATHS -------------------------------------------------#
AMAZON_REVIEWS_DATASETS_PATH = DATA_PATH / "reviews" / "amazon"
AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON = (
    AMAZON_REVIEWS_DATASETS_PATH / "reviews_Automotive.json"
)
AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON = (
    AMAZON_REVIEWS_DATASETS_PATH / "reviews_Apps_for_Android.json"
)
AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON = (
    AMAZON_REVIEWS_DATASETS_PATH / "reviews_Amazon_Instant_Video.json"
)
AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON = (
    AMAZON_REVIEWS_DATASETS_PATH / "reviews_Cell_Phones_and_Accessories.json"
)

# --------------------------------------------- BING LIU REVIEWS ASPECT-BASED DATASETS-------------------------------- #
BING_LIU_DATASETS_PATH = DATA_PATH / "aspects" / "bing_liu" / "json"
BING_LIU_POWERSHOT = BING_LIU_DATASETS_PATH / "Canon PowerShot SD500.json"
BING_LIU_S100 = BING_LIU_DATASETS_PATH / "Canon S100.json"
BING_LIU_DIAPER_CHAMP = BING_LIU_DATASETS_PATH / "Diaper Champ.json"
BING_LIU_HITACHI = BING_LIU_DATASETS_PATH / "Hitachi router.json"
BING_LIU_IPOD = BING_LIU_DATASETS_PATH / "ipod.json"
BING_LIU_LINKSYS_ROUTER = BING_LIU_DATASETS_PATH / "Linksys Router.json"
BING_LIU_MICRO_MP3 = BING_LIU_DATASETS_PATH / "MicroMP3.json"
BING_LIU_NOKIA_6600 = BING_LIU_DATASETS_PATH / "Nokia 6600.json"
BING_LIU_NORTON = BING_LIU_DATASETS_PATH / "norton.json"

ALL_BING_LIU_ASPECTS_PATH = (
    ROOT_PATH.parent / "aspects" / "data" / "aspects" / "bing_liu"
)
BING_LIU_9_PRODUCTS_PATH = ALL_BING_LIU_ASPECTS_PATH / "Reviews-9-products"
BING_LIU_CUSTOMER_REVIEWS_PATH = ALL_BING_LIU_ASPECTS_PATH / "customer review data"
BING_LIU_CUSTOMER_REVIEWS_3_DOMAINS_PATH = (
    ALL_BING_LIU_ASPECTS_PATH / "CustomerReviews -3 domains (IJCAI2015)"
)

BING_LIU_ASPECT_DATASETS_PATHS = (
    list(BING_LIU_9_PRODUCTS_PATH.glob("*.txt"))
    + list(BING_LIU_CUSTOMER_REVIEWS_PATH.glob("*.txt"))
    + list(BING_LIU_CUSTOMER_REVIEWS_3_DOMAINS_PATH.glob("*.txt"))
)

BING_LIU_BIO_DATASET = ALL_BING_LIU_ASPECTS_PATH / "bio_tags"

# --------------------------------------------- SEMEVAL DATASETS ----------------------------------------------------- #
SEMEVAL_DATASETS = DATA_PATH / "semeval"
SEMEVAL_DATASETS_2014 = SEMEVAL_DATASETS / "2014"
SEMEVAL_DATASETS_2016 = SEMEVAL_DATASETS / "2016"
SEMEVAL_RESTAURANTS_TRAIN_XML = SEMEVAL_DATASETS_2014 / "Restaurants_Train.xml"
SEMEVAL_RESTAURANTS_TEST_XML = (
    SEMEVAL_DATASETS_2014 / "Restaurants_Test_Data_phaseB.xml"
)
SEMEVAL_LAPTOPS_TRAIN_XML = SEMEVAL_DATASETS_2014 / "Laptops_Train.xml"
SEMEVAL_LAPTOPS_TEST_XML = SEMEVAL_DATASETS_2014 / "Laptops_Test_Data_phaseB.xml"

# --------------------------------------------- SENTIMENT MODELS ----------------------------------------------------- #
SENTIMENT_MODELS_PATH = DATA_PATH / "models" / "sentiment"
SENTIMENT_MODEL_PROD = (
    SENTIMENT_MODELS_PATH
    / "Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-10-domains.pkl"
)

# smaller model as default - useful for testing
SENTIMENT_MODEL_TESTS = (
    SENTIMENT_MODELS_PATH
    / "Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-reviews_Apps_for_Android-500000-balanced.pkl"
)

SENTIMENT_DOCKER_URL = "http://localhost:5002/api/sentiment"

# --------------------------------------------- RST  ----------------------------------------------------------------- #

RST_PARSER_DOCKER_URL = "http://localhost:5000/api/rst/parse"
RETRIES_LIMIT = 100

# --------------------------------------------- ASPECT MODELS -------------------------------------------------------- #

ASPECT_EXTRACTION_TRAIN_DATASET = (
    DATA_PATH / "aspects" / "merged-electronic-aspects-uni-tag.conll"
)

ASPECT_EXTRACTOR_DOCKER_URL = "http://localhost:5001/api/aspects"

# --------------------------------------------- EMBEDDINGS ----------------------------------------------------------- #


WORD_EMBEDDING_GLOVE_42B = DATA_PATH / "embedding" / "glove.42B.300d.txt"
WORD_EMBEDDING_GLOVE_42B_VOCAB = DATA_PATH / "embedding" / "glove.42B.300d.vocab.pkl"

# --------------------------------------------- CONCEPTNETS ---------------------------------------------------------- #
CONCEPTNET_CSV_EN_PATH = DATA_PATH / "conceptnet" / "conceptnet-5.7.0-assertions.en.csv"
CONCEPTNET_CSV_PL_PATH = DATA_PATH / "conceptnet" / "conceptnet-5.7.0-assertions.pl.csv"

CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_NO_SYNONYMS_EN_PATH = (
    DATA_PATH / "conceptnet" / "conceptnet-5.7.0-assertions.en.v3.gt"
)

CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITH_SYNONYMS_EN_PATH = (
    DATA_PATH / "conceptnet" / "5.7.0-hierarchical.with.synonyms.en.gt"
)
CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITHOUT_SYNONYMS_EN_PATH = (
    DATA_PATH / "conceptnet" / "5.7.0-hierarchical.without.synonyms.en.gt"
)
CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITH_SYNONYMS_AND_RELATED_TO_EN_PATH = (
    DATA_PATH / "conceptnet" / "5.7.0-hierarchical.with.synonyms.and.related.to.en.gt"
)

CONCEPTNET_GRAPH_TOOL_ALL_RELATIONS_WITH_SYNONYMS_EN_PATH = (
    DATA_PATH / "conceptnet" / "5.7.0-all-relations.with.synonyms.en.gt"
)
CONCEPTNET_GRAPH_TOOL_ALL_RELATIONS_WITHOUT_SYNONYMS_EN_PATH = (
    DATA_PATH / "conceptnet" / "5.7.0-all-relations.without.synonyms.en.gt"
)
# FIXME update pkl with 5.7.0 dump of assertions - only polish and english are important
CONCEPTNET_IO_PKL = DATA_PATH / "conceptnet" / "conceptnet_io.pkl"

# sentic net conceptnet
# Do we use sentic conceptnet based entities in aspect extraction procedure.
# We use sentic.net as data source here.
SENTIC_ASPECTS = True
SENTIC_EXACT_MATCH_CONCEPTS = True

# conceptnet io
# Do we use ConceptNet based entities in aspect extraction procedure.
# We use conceptnet.io as data source here.
CONCEPTNET_IO_ASPECTS = True
CONCEPTNET_IO_LANG = u"en"
CONCEPTNET_IO_RELATIONS = [
    u"LocatedNear",
    u"HasA",
    u"PartOf",
    u"MadeOf",
    u"IsA",
    u"InheritsFrom",
    u"Synonym",
]

# --------------------------------------------- OTHER SETTINGS ----------------------------------------------------- #
NER_TYPES = [u"PERSON", u"GPE", u"ORG", u"PRODUCT", u"FAC", u"LOC"]

# serialization steps
ASPECT_EXTRACTION_SERIALIZATION_STEP = 10000

# sample trees
SAMPLE_TREE_1 = DATA_PATH / "sample_trees" / "1.tree"
SAMPLE_TREE_177 = DATA_PATH / "sample_trees" / "177.tree"
SAMPLE_TREE_189 = DATA_PATH / "sample_trees" / "189.tree"

ML_GUSE_MODEL_2_LITE_PATH = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"

DISCOURSE_TREE_LEAF_PATTERN = "(?<=_!).*(?=!_)"
