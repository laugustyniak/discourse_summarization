import logging
from glob import glob
from os.path import join, basename

from run import AspectAnalysisSystem

reviews = glob('aspects/data/reviews/*.json')

reviews = [review for review in reviews if 'crappy_example' not in review]

log = logging.getLogger(__name__)

for review in reviews:
    input_folder_name = basename(review).replace('.json', '')
    log.info('Review file {}'.format(review))
    AAS = AspectAnalysisSystem(input_path=review,
                               output_path=join('results', input_folder_name),
                               gold_standard_path=None,
                               analysis_results_path=None,
                               jobs=7,
                               batch_size=500)
    AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
