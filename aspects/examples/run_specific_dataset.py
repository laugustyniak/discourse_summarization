from aspects.embeddings.graph.conceptnet_hierarchies_check import prepare_hierarchies_neighborhood
from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings

DATASET_PATH = settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON

# TODO: better parametrization for experiments and domain's evaluation
# TODO: params: amazon domain, set of conceptnet relations, our vs gerani arrg generation
# aspect_analysis_gerani = AspectAnalysis(
#     input_path=DATASET_PATH.as_posix(),
#     output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
#     experiment_name='gerani',
#     jobs=2,
#     batch_size=100,
# )
# aspect_analysis_gerani.gerani_pipeline()
# prepare_hierarchies_neighborhood(reviews_path=aspect_analysis_gerani.output_path)

aspect_analysis_our = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    experiment_name='our',
    jobs=2,
    batch_size=100,
)
aspect_analysis_our.our_pipeline()
prepare_hierarchies_neighborhood(reviews_path=aspect_analysis_our.output_path)

# TODO: generate graph for each variation of params, use notebook's code
# sns.lineplot(x=df.shortest_distance_aspect_graph, y=df.shortest_distance_conceptnet)
