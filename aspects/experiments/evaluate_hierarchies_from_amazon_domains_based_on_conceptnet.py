import pandas as pd
import seaborn as sns
from tqdm import tqdm

from aspects.data.conceptnet.graphs import CONCEPTNET_GRAPH_TOOL_GRAPHS
from aspects.graph.graph_tool.conceptnet_hierarchies_check import prepare_hierarchies_neighborhood
from aspects.graph.graph_tool.utils import VALUES_TO_SKIP
from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings

sns.set(color_codes=True)
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

# TODO: generate graph for each variation of params, use notebook's code
aspect_analysis_our = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    experiment_name='our',
    jobs=2,
    batch_size=100,
)
aspect_analysis_our.our_pipeline()

for conceptnet_graph_path in tqdm(CONCEPTNET_GRAPH_TOOL_GRAPHS, desc='Conceptnet graph analysis...'):
    prepare_hierarchies_neighborhood(
        reviews_path=aspect_analysis_our.output_path,
        conceptnet_graph_path=conceptnet_graph_path
    )
    df = pd.read_pickle(aspect_analysis_our.paths.conceptnet_hierarchy_neighborhood)
    df = df[~(
            (df.shortest_distance_aspect_graph.isin(VALUES_TO_SKIP)) |
            (df.shortest_distance_conceptnet.isin(VALUES_TO_SKIP))
    )]
    df['shortest_paths_differences'] = df.shortest_distance_conceptnet - df.shortest_distance_aspect_graph
    df.drop_duplicates(subset=['aspect_1', 'aspect_2'])
    sns_plot = sns.lineplot(x=df.shortest_distance_aspect_graph, y=df.shortest_distance_conceptnet)
    sns_plot.savefig(str(aspect_analysis_our / f"shortest_paths_correlation_{conceptnet_graph_path.stem}.png"))
