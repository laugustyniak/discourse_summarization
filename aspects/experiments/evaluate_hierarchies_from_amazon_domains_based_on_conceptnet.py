import logging

import pandas as pd
import seaborn as sns
from tqdm import tqdm

from aspects.data.conceptnet.graphs import CONCEPTNET_GRAPH_TOOL_GRAPHS
from aspects.graph.graph_tool.conceptnet_hierarchies_check import prepare_hierarchies_neighborhood
from aspects.graph.graph_tool.utils import VALUES_TO_SKIP
from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings

sns.set(color_codes=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

datasets = [
    (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, None),
    (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, None),
    (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, 50000),
    (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, 50000),
]

for dataset_path, max_reviews in tqdm(datasets, desc='Amazon datasets processing...'):
    for experiment_name in ['our', 'gerani']:
        aspect_analysis_our = AspectAnalysis(
            input_path=dataset_path.as_posix(),
            output_path=settings.DEFAULT_OUTPUT_PATH / dataset_path.stem,
            experiment_name=experiment_name,
            jobs=16,
            batch_size=500,
            max_docs=max_reviews
        )
        if experiment_name in ['our']:
            aspect_analysis_our.our_pipeline()
        elif experiment_name in ['our']:
            aspect_analysis_our.gerani_pipeline()
        else:
            raise Exception('Wrong experiment type')

        for conceptnet_graph_path in tqdm(CONCEPTNET_GRAPH_TOOL_GRAPHS, desc='Conceptnet graph analysis...'):
            prepare_hierarchies_neighborhood(
                experiments_path=aspect_analysis_our.paths,
                conceptnet_graph_path=conceptnet_graph_path
            )

            df = pd.read_pickle(aspect_analysis_our.paths.conceptnet_hierarchy_neighborhood)
            logger.info(f'Shortest Paths pairs - data frame: {len(df)}')
            df = df[~(
                    (df.shortest_distance_aspect_graph.isin(VALUES_TO_SKIP)) |
                    (df.shortest_distance_conceptnet.isin(VALUES_TO_SKIP))
            )]
            df.drop_duplicates(subset=['aspect_1', 'aspect_2'])
            logger.info(f'Shortest Paths pairs - data frame, without no paths and duplicates: {len(df)}')

            sns_plot = sns.lineplot(x=df.shortest_distance_aspect_graph, y=df.shortest_distance_conceptnet)
            png_file_path = (
                    aspect_analysis_our.paths.experiment_path /
                    f"shortest_paths_correlation_{conceptnet_graph_path.stem}.png"
            ).as_posix()
            logger.info(f'Shortest Paths correlation figure will be saved in {png_file_path}')
            sns_plot.figure.savefig(png_file_path)
