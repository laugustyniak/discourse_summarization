import logging
import multiprocessing
from glob import glob
from os.path import join, basename

from run import AspectAnalysis

reviews = glob('aspects/data/reviews/*.json')

reviews = [review for review in reviews if 'crappy_example' not in review]

log = logging.getLogger(__name__)

for review in reviews:
    input_folder_name = basename(review).replace('.json', '')
    log.info('Review file {}'.format(review))

    N_JOBS = multiprocessing.cpu_count() / 2

    AAS = AspectAnalysis(
        input_path=review,
        output_path=join('results', basename(review).replace('.json', '')),
        gold_standard_path=None,
        analysis_results_path=None,
        jobs=N_JOBS,
        batch_size=50,
        neutral_sent=False,
        cycle_in_relations=False,
        filter_gerani=False,
        aht_gerani=False,
    )

    AAS.run()

# python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
