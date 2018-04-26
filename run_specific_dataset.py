import logging
import multiprocessing
from os.path import join, basename

from aspects.utilities import settings
from run import AspectAnalysisSystem

log = logging.getLogger(__name__)

dataset_path = settings.BING_LIU_LINKSYS_ROUTER.as_posix()

AAS = AspectAnalysisSystem(
    input_path=dataset_path,
    output_path=join('results', basename(dataset_path).replace('.json', '')),
    gold_standard_path=None,
    analysis_results_path=None,
    jobs=multiprocessing.cpu_count() - 1,
    batch_size=50,
    # max_docs=15000,
)

AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
