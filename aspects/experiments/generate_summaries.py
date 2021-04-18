import logging
from typing import Union

import click
import mlflow
import seaborn as sns
from tqdm import tqdm

from aspects.experiments import experiment_name_enum
from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings
from aspects.utilities.settings import setup_mlflow

sns.set(color_codes=True)
setup_mlflow()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASETS = [
    (settings.AMAZON_REVIEWS_AUTOMOTIVE_DATASET_JSON, 25_000),
    (settings.AMAZON_REVIEWS_AMAZON_INSTANT_VIDEO_DATASET_JSON, 25_000),
    (settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON, 50_000),
    (settings.AMAZON_REVIEWS_APPS_FOR_ANDROID_DATASET_JSON, None),
]

EXPERIMENTS = [
    experiment_name_enum.GERANI,
    experiment_name_enum.OUR,
    experiment_name_enum.OUR_TOP_1_RULES,
    experiment_name_enum.OUR_TOP_5_RULES,
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
@click.option("--experiment-id", default=8, help="name of experiment for mlflow")
def main(
    n_jobs: int,
    batch_size: int,
    aht_max_number_of_nodes: int,
    alpha_coefficient: float,
    experiment_id: Union[str, int],
):
    for dataset_path, max_reviews in tqdm(
        DATASETS, desc="Amazon datasets processing..."
    ):
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{dataset_path.stem}-{max_reviews}",
        ):
            for experiment_name in EXPERIMENTS:

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
                    aspect_analysis.our_pipeline_top_n_rules_per_discourse_tree(
                        top_n=1, use_aspect_clustering=False
                    )
                elif experiment_name == experiment_name_enum.OUR_TOP_5_RULES:
                    aspect_analysis.our_pipeline_top_n_rules_per_discourse_tree(
                        top_n=5, use_aspect_clustering=False
                    )
                else:
                    raise Exception("Wrong experiment type")

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_name", dataset_path.stem)
        mlflow.log_param("max_docs", max_reviews)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("aht_max_number_of_nodes", aht_max_number_of_nodes)
        mlflow.log_param("alpha_coefficient", alpha_coefficient)


if __name__ == "__main__":
    main()
