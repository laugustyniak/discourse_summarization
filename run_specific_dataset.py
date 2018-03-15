import logging
import multiprocessing

from aspects.utilities import settings
from run import AspectAnalysisSystem

log = logging.getLogger(__name__)

AAS = AspectAnalysisSystem(input_path=str(settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON),
                           output_path='results',
                           gold_standard_path=None,
                           analysis_results_path=None,
                           jobs=multiprocessing.cpu_count() - 1,
                           )
AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23