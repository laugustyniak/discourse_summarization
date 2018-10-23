import logging
import multiprocessing
from os.path import join, basename

from aspects.utilities import settings
from run import AspectAnalysisSystem

log = logging.getLogger(__name__)

# DATASET_PATH = settings.BING_LIU_IPOD.as_posix()
DATASET_PATH = settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON.as_posix()
N_JOBS = multiprocessing.cpu_count() - 1

AAS = AspectAnalysisSystem(
    input_path=DATASET_PATH,
    output_path=join('results', basename(DATASET_PATH).replace('.json', '')),
    gold_standard_path=None,
    analysis_results_path=None,
    jobs=N_JOBS,
    batch_size=50,
    neutral_sent=True,
    cycle_in_relations=False,
    filter_gerani=False,
    aht_gerani=False,
    # max_docs=15000,
)

AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
