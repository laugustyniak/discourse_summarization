from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings

DATASET_PATH = settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON

aspect_analysis = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    jobs=2,
    batch_size=100,
    max_docs=50000
)

aspect_analysis.gerani_pipeline()
