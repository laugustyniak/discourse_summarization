import logging
from typing import Union

import click
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
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

datasets = [
    # (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, 50),
    (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, 10**5),
    # (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, 5001),
    # (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, 50001),
    # (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, 50001),
    # (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, 50001),
    # (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, 50001),
    # (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, None),
    # (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, None), # not all trees processed
    # (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, None),
    # (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, None),
    # (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, 50000),
]


@click.command()
@click.option(
    "--n-jobs", default=1, help="Number of concurrent job for parts of pipeline."
)
@click.option(
    "--batch-size",
    default=100,
    help="Number of examples that could we process by one process",
)
@click.option("--aht_max_number_of_nodes", default=50, help="Max nodes for AHT")
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
        for experiment_name in ["our", "gerani"]:

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

                if experiment_name in ["our"]:
                    aspect_analysis.our_pipeline()
                elif experiment_name in ["gerani"]:
                    aspect_analysis.gerani_pipeline()
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

                            plt.figure()
                            sns_plot = sns.lineplot(
                                x=df.shortest_distance_aspect_graph,
                                y=df.shortest_distance_conceptnet,
                            )
                            logger.info(
                                f"Shortest Paths correlation figure will be saved in {png_file_path}"
                            )
                            sns_plot.figure.savefig(png_file_path.as_posix())
                            plt.close()

                            mlflow.log_artifact(png_file_path.as_posix())


if __name__ == "__main__":
    main()
