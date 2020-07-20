from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings

DATASET_PATH = settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON

aspect_analysis_gerani = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    experiment_name='gerani',
    jobs=2,
    batch_size=100,
)
aspect_analysis_gerani.gerani_pipeline()

aspect_analysis_our = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    experiment_name='our',
    jobs=2,
    batch_size=100,
)
aspect_analysis_our.our_pipeline()
