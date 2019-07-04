import logging
import multiprocessing
from os.path import join, basename

from aspects.utilities import settings
from run import AspectAnalysisSystem

log = logging.getLogger(__name__)

DATASET_PATH = settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON
N_JOBS = multiprocessing.cpu_count() - 2

OXYGEN_RESULTS_PATH = '/datasets/sentiment/aspects/results'

AAS = AspectAnalysisSystem(
    input_path=DATASET_PATH.as_posix(),
    output_path=join('results', DATASET_PATH.stem),
    # output_path=join(OXYGEN_RESULTS_PATH, basename(DATASET_PATH).replace('.json', '')),
    gold_standard_path=None,
    analysis_results_path=None,
    jobs=N_JOBS,
    batch_size=1000,
    neutral_sent=True,
    cycle_in_relations=False,
    filter_gerani=False,
    aht_gerani=False,
    # max_docs=20000,
)

AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
