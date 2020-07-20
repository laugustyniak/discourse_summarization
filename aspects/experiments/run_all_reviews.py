import logging
from glob import glob
from os.path import basename

from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities.settings import DEFAULT_OUTPUT_PATH

# TODO: path where to look for json files
reviews = glob('aspects/data/reviews/*.json')

reviews = [review for review in reviews if 'crappy_example' not in review]

log = logging.getLogger(__name__)

for review in reviews:
    input_folder_name = basename(review).replace('.json', '')
    log.info('Review file {}'.format(review))

    aspect_analysis = AspectAnalysis(
        input_path=review,
        output_path=DEFAULT_OUTPUT_PATH / basename(review).replace('.json', ''),
        batch_size=50,
    )

    aspect_analysis.gerani_pipeline()
