import logging
from glob import glob

from aspects.analysis.run import AspectAnalysisSystem

reviews = glob('data/reviews/*.json')

log = logging.getLogger(__name__)

for review in reviews:
    log.info('Review file {}'.format(review))
    AAS = AspectAnalysisSystem(input_path=review,
                               output_path='results',
                               gold_standard_path=None,
                               jobs=7)
    AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
