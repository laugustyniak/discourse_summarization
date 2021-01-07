import pandas as pd
import logging
from typing import Union

import click
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from scipy.stats import pearsonr

from aspects.experiments import experiment_name_enum
from tqdm import tqdm

from aspects.data.conceptnet.graphs import CONCEPTNET_GRAPH_TOOL_GRAPHS
from aspects.graph.graph_tool.conceptnet_hierarchies_check import (
    prepare_hierarchies_neighborhood,
)
from aspects.graph.graph_tool.utils import VALUES_TO_SKIP
from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings
from aspects.utilities.settings import setup_mlflow

sns.set(color_codes=True)
setup_mlflow()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

N_CALLS = 25_000
N_CALLS_2 = 100_000

datasets = [
    (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, N_CALLS),
    (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, N_CALLS),
    (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, N_CALLS),
    (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, N_CALLS),
    (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, N_CALLS_2),
    (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, N_CALLS_2),
    (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, N_CALLS_2),
    (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, N_CALLS_2),
    (settings.EVENT_REGISTRY_BREXIT_NEWS_WITH_BODY_LARGE, 10000),
    (settings.EVENT_REGISTRY_BREXIT_NEWS_LARGE, None),
]


@click.command()
@click.option(
    "--n-jobs",
    default=20,
    help="Number of concurrent job for parts of pipeline. "
    "-1 means all cores will be used.",
)
@click.option(
    "--batch-size",
    default=50,
    help="Number of examples that could we process by one process",
)
@click.option("--aht_max_number_of_nodes", default=100, help="Max nodes for AHT")
@click.option(
    "--alpha_coefficient", default=0.5, help="Alpha coefficient for moi calculation"
)
@click.option("--experiment-id", default=3, help="name of experiment for mlflow")
@click.option(
    "--overwrite-neighborhood/--no-overwrite-neighborhood",
    default=True,
    help="Calculate neighborhoods once again",
)
@click.option(
    "--filter-graphs-to-intersected-vertices/--no-filter-graphs-to-intersected-vertices",
    default=True,
    type=bool,
)
def main(
    n_jobs: int,
    batch_size: int,
    aht_max_number_of_nodes: int,
    alpha_coefficient: float,
    experiment_id: Union[str, int],
    overwrite_neighborhood: bool,
    filter_graphs_to_intersected_vertices: bool,
):
    filter_graphs_to_intersected_vertices = bool(filter_graphs_to_intersected_vertices)
    for dataset_path, max_reviews in tqdm(
        datasets, desc="Amazon datasets processing..."
    ):
        for experiment_name in [
            experiment_name_enum.GERANI,
            experiment_name_enum.OUR,
            experiment_name_enum.OUR_TOP_1_RULES,
        ]:

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{experiment_name}-{dataset_path.stem}-{max_reviews}",
            ):
                mlflow.log_param("experiment_name", experiment_name)

                aspect_analysis = AspectAnalysis(
                    input_path=dataset_path.as_posix(),
                    output_path=settings.DEFAULT_OUTPUT_PATH / dataset_path.stem,
                    experiment_name=experiment_name,
                    jobs=n_jobs,
                    batch_size=batch_size,
                    max_docs=max_reviews,
                    aht_max_number_of_nodes=aht_max_number_of_nodes,
                    alpha_coefficient=alpha_coefficient,
                )

                if experiment_name == experiment_name_enum.OUR:
                    aspect_analysis.our_pipeline()
                elif experiment_name == experiment_name_enum.GERANI:
                    aspect_analysis.gerani_pipeline()
                elif experiment_name == experiment_name_enum.OUR_TOP_1_RULES:
                    aspect_analysis.our_pipeline_top_n_rules_per_discourse_tree()
                else:
                    raise Exception("Wrong experiment type")

                for conceptnet_graph_path in tqdm(
                    CONCEPTNET_GRAPH_TOOL_GRAPHS, desc="Conceptnet graph analysis..."
                ):

                    with mlflow.start_run(
                        experiment_id=experiment_id,
                        run_name=conceptnet_graph_path.stem,
                        nested=True,
                        # run_id=f'{experiment_id}-{conceptnet_graph_path.stem}'
                    ) as run_conceptnet:

                        mlflow.log_param("dataset_path", dataset_path)
                        mlflow.log_param("dataset_name", dataset_path.stem)
                        mlflow.log_param("method", experiment_name)
                        mlflow.log_param("max_docs", max_reviews)
                        mlflow.log_param("batch_size", batch_size)
                        mlflow.log_param("n_jobs", n_jobs)
                        mlflow.log_param("conceptnet_graph_path", conceptnet_graph_path)
                        mlflow.log_param(
                            "conceptnet_graph_name", conceptnet_graph_path.stem
                        )
                        mlflow.log_param(
                            "aht_max_number_of_nodes", aht_max_number_of_nodes
                        )
                        mlflow.log_param("alpha_coefficient", alpha_coefficient)

                        png_file_path = (
                            aspect_analysis.paths.experiment_path
                            / f"shortest_paths_correlation_{conceptnet_graph_path.stem}.png"
                        )

                        if png_file_path.exists() and not overwrite_neighborhood:
                            logger.info(
                                f"{png_file_path.as_posix()} has already exist, skipping to the next setting."
                            )
                            mlflow.log_artifact(png_file_path.as_posix())
                        else:
                            df = prepare_hierarchies_neighborhood(
                                experiments_path=aspect_analysis.paths,
                                conceptnet_graph_path=conceptnet_graph_path,
                                filter_graphs_to_intersected_vertices=filter_graphs_to_intersected_vertices,
                            )

                            logger.info(f"Shortest Paths pairs - data frame: {len(df)}")
                            df = df[
                                ~(
                                    (
                                        df.shortest_distance_aspect_graph.isin(
                                            VALUES_TO_SKIP
                                        )
                                    )
                                    | (
                                        df.shortest_distance_conceptnet.isin(
                                            VALUES_TO_SKIP
                                        )
                                    )
                                )
                            ]
                            df.drop_duplicates(subset=["aspect_1", "aspect_2"])
                            mlflow.log_metric("number_of_shortest_paths", len(df))
                            logger.info(
                                f"Shortest Paths pairs - data frame, without no paths and duplicates: {len(df)}"
                            )

                            mlflow.log_dict(
                                pd.DataFrame(
                                    df.shortest_distance_aspect_graph.value_counts()
                                ).to_dict(orient="index"),
                                "shortest_distance_aspect_graph_distribution.json",
                            )

                            mlflow.log_dict(
                                pd.DataFrame(
                                    df.shortest_distance_conceptnet.value_counts()
                                ).to_dict(orient="index"),
                                "shortest_distance_conceptnet_distribution.json",
                            )

                            df = df[df.shortest_distance_aspect_graph <= 6]

                            matplotlib.rc_file_defaults()
                            ax1 = sns.set_style(style=None, rc=None)
                            fig, ax1 = plt.subplots()
                            sns_plot = sns.lineplot(
                                x=df.shortest_distance_aspect_graph,
                                y=df.shortest_distance_conceptnet,
                                ax=ax1,
                            )
                            ax2 = ax1.twinx()
                            df_aspect_graph_distance_distribution = pd.DataFrame(
                                df.shortest_distance_aspect_graph.value_counts()
                            )
                            df_aspect_graph_distance_distribution.reset_index(
                                inplace=True
                            )
                            df_aspect_graph_distance_distribution.sort_values(
                                by="index", inplace=True
                            )
                            sns.barplot(
                                x=df_aspect_graph_distance_distribution["index"],
                                y=df_aspect_graph_distance_distribution.shortest_distance_aspect_graph,
                                alpha=0.5,
                                ax=ax2,
                            )
                            logger.info(
                                f"Shortest Paths correlation figure will be saved in {png_file_path}"
                            )
                            pearson_values = pearsonr(
                                x=df.shortest_distance_aspect_graph.tolist(),
                                y=df.shortest_distance_conceptnet.tolist(),
                            )
                            mlflow.log_metrics(
                                {
                                    "pearsonr": pearson_values[0],
                                    "pearsonr_p-value": pearson_values[1],
                                }
                            )
                            sns_plot.figure.savefig(png_file_path.as_posix())
                            plt.close()

                            mlflow.log_artifact(png_file_path.as_posix())


if __name__ == "__main__":
    main()
