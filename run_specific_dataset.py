from os.path import join

from aspects.utilities import settings
from run import AspectAnalysisSystem

DATASET_PATH = settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON

AAS = AspectAnalysisSystem(
    input_path=DATASET_PATH.as_posix(),
    output_path=join('results', DATASET_PATH.stem),
    analysis_results_path=None,
    jobs=7,
    batch_size=1000,
    neutral_sent=True,
    cycle_in_relations=False,
    filter_gerani=False,
    aht_gerani=False,
)

AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
