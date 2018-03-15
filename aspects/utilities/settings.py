import pathlib

ROOT_PATH = pathlib.Path(__file__).absolute().parent.parent

# AMAZON DATASETS PATHS
AMAZON_REVIEWS_DATASETS_PATH = ROOT_PATH / 'data' / 'reviews' / 'amazon'
AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_GZ = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Apps_for_Android.json.gz'
AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Apps_for_Android.json'
AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_GZ = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Amazon_Instant_Video.json.gz'
AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON = AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Amazon_Instant_Video.json'
AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_GZ = \
    AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Cell_Phones_and_Accessories.json.gz'
AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON = \
    AMAZON_REVIEWS_DATASETS_PATH / 'reviews_Cell_Phones_and_Accessories.json'

# SENITMENT MODELS
SENTIMENT_MODELS_PATH = ROOT_PATH / 'data' / 'models'
SENTIMENT_MODEL_PROD = SENTIMENT_MODELS_PATH / \
                       'Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5-10-domains.pkl'
# smaller model as default - useful for testing
SENTIMENT_MODEL_TESTS = SENTIMENT_MODELS_PATH / 'Pipeline-LogisticRegression-CountVectorizer-n_grams_1_2-stars-1-3-5' \
                                                '-reviews_Apps_for_Android-500000-balanced.pkl'
