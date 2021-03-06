import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures.process import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from os.path import basename
from pathlib import Path
from typing import Callable, List, Sequence, Union

import mlflow
import networkx as nx
import pandas as pd
from more_itertools import flatten
from tqdm import tqdm

from aspects.analysis.gerani_graph_analysis import (
    extend_graph_nodes_with_sentiments_and_weights,
    gerani_paper_arrg_to_aht,
    our_paper_arrg_to_aht,
)
from aspects.aspects import rule_filters
from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.aspects.aspects_graph_builder import (
    Aspect2AspectGraph,
    sort_networkx_attributes,
)
from aspects.data_io import serializer
from aspects.rst.extractors import (
    extract_discourse_tree,
    extract_discourse_tree_with_ids_only,
    extract_rules,
)
from aspects.sentiment.simple_textblob import analyze
from aspects.utilities import pandas_utils
from aspects.utilities.data_paths import ExperimentPaths
from aspects.utilities.settings import setup_mlflow
from aspects.visualization.drawing import draw_tree

setup_mlflow()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s;%(filename)s:%(lineno)s;" "%(funcName)s();%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="analyzer.log",
)

ASPECTS_TO_SKIP = frozenset([
    "fyi",
    "un",
    "dynamat",
    "yz",
    "que es de buena calidad",
    "love",
    "nice app",
    "inc",
    "ra",
    "st. patrick 's",
    "swan slimline",
    "ty",
    "ewing",
    "ozzy",
    "swan slimline",
    "aziz",
    "xxl",
    "rzr",
])


class AspectAnalysis:
    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        experiment_name: str = "",
        jobs: int = None,
        sent_model_path: Union[str, Path] = None,
        batch_size: int = None,
        max_docs: int = None,
        alpha_coefficient: float = 0.5,
        aht_max_number_of_nodes: int = 50,
        min_freq_of_aspects: int = 1,
    ):
        self.max_docs = max_docs
        mlflow.log_param("max_docs", max_docs)
        self.batch_size = batch_size
        mlflow.log_param("batch_size", max_docs)
        self.input_file_path = input_path
        if self.max_docs is not None:
            self.output_path = f"{str(output_path)}-{self.max_docs}-docs"
        else:
            self.output_path = output_path
        self.experiment_name = experiment_name
        self.paths = ExperimentPaths(input_path, self.output_path, experiment_name)
        self.sent_model_path = sent_model_path
        mlflow.log_param("sent_model_path", sent_model_path)

        # number of all processes
        if jobs is None or jobs == -1:
            self.jobs = multiprocessing.cpu_count() - 1
        else:
            self.jobs = jobs
        mlflow.log_param("n_jobs", self.jobs)

        self.alpha_coefficient = alpha_coefficient
        mlflow.log_param("alpha_coefficient", alpha_coefficient)
        self.aht_max_number_of_nodes = aht_max_number_of_nodes
        mlflow.log_param("aht_max_number_of_nodes", aht_max_number_of_nodes)
        self.min_freq_of_aspects = min_freq_of_aspects
        mlflow.log_param("min_freq_of_aspects", min_freq_of_aspects)

    def parallelized_extraction(
        self, elements: Sequence, fn: Callable, desc: str = "Running in parallel"
    ) -> List:
        with ProcessPoolExecutor(self.jobs) as pool:
            return list(
                tqdm(
                    pool.map(fn, elements, chunksize=self.batch_size),
                    total=len(elements),
                    desc=desc,
                )
            )

    def extract_discourse_trees(self) -> pd.DataFrame:
        if self.paths.discourse_trees_df.exists():
            logging.info("Discourse trees loading.")
            return pd.read_pickle(self.paths.discourse_trees_df)
        else:
            print(f"{self.input_file_path} will be loaded!")
            f_extension = basename(self.input_file_path).split(".")[-1]

            if f_extension in ["json"]:
                with open(self.input_file_path, "r") as json_file:
                    df = pd.DataFrame(json.load(json_file).values(), columns=["text"])
            elif f_extension in ["csv", "txt"]:
                df = pd.read_csv(self.input_file_path, header=None)
                df.columns = ["text"]
            else:
                raise Exception("Wrong file type! It must be [json, txt or csv]")

            if self.max_docs is not None:
                df = df.head(self.max_docs)

            mlflow.log_param(
                "discourse_parsing_start_time",
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            )
            df["discourse_tree"] = self.parallelized_extraction(
                df.text.tolist(), extract_discourse_tree, "Discourse trees parsing"
            )
            mlflow.log_param(
                "discourse_parsing_end_time",
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            )

            n_docs = len(df)
            df.dropna(subset=["discourse_tree"], inplace=True)
            logging.info(
                f"{n_docs - len(df)} discourse tree has been parser with errors and we skip them."
            )

            assert not df.empty, "No trees to process!"
            assert (
                df[df.edus.apply(lambda l: len(l) > 0)].shape[0] > df.shape[0] / 2
            ), "Probably to many RST errors, please check to discourse trees!"

            self.discourse_trees_df_checkpoint(df)

        return df

    def extract_discourse_trees_ids_only(self, df: pd.DataFrame) -> pd.DataFrame:
        if "discourse_tree_ids_only" in df.columns:
            return df

        df["discourse_tree_ids_only"], df["edus"] = tuple(
            zip(
                *self.parallelized_extraction(
                    df.discourse_tree.tolist(),
                    extract_discourse_tree_with_ids_only,
                    "Discourse trees parsing to idx only",
                )
            )
        )
        self.discourse_trees_df_checkpoint(df)

        return df

    def discourse_trees_df_checkpoint(self, df: pd.DataFrame):
        logging.info(f"Discourse data frame - saving.")
        df.to_pickle(self.paths.discourse_trees_df)
        logging.info(f"Discourse data frame - saved.")

    def extract_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sentiment" in df.columns:
            logging.info(
                "Sentiments have been already extracted. Passing to the next step."
            )
            return df

        pandas_utils.assert_columns(df, "edus")
        df["sentiment"] = self.parallelized_extraction(
            df.edus.tolist(), analyze, "Sentiment extracting"
        )
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        if "aspects" in df.columns:
            logging.info(
                "Aspects have been already extracted. Passing to the next step."
            )
            return df

        pandas_utils.assert_columns(df, "edus")

        extractor = AspectExtractor()
        df["aspects"] = self.parallelized_extraction(
            df.edus.tolist(), extractor.extract_batch, "Aspects extracting"
        )
        self.discourse_trees_df_checkpoint(df)

        # df["concepts"] = self.parallelized_extraction(
        #     df.aspects.tolist(), extractor.extract_concepts_batch, "Concepts extracting"
        # )
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_edu_rhetorical_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        # if "rules" in df.columns:
        #     return df

        pandas_utils.assert_columns(df, "discourse_tree_ids_only")

        df["rules"] = self.parallelized_extraction(
            df.discourse_tree_ids_only.tolist(), extract_rules, "Extracting rules"
        )

        self.discourse_trees_df_checkpoint(df)

        return df

    def build_aspect_to_aspect_graph(
        self,
        df: pd.DataFrame,
        filter_relation_fn: Callable = None,
        aspects_to_skip=None,
    ):
        builder = Aspect2AspectGraph(aspects_to_skip=aspects_to_skip)
        graph = builder.build(
            discourse_tree_df=df, filter_relation_fn=filter_relation_fn
        )
        serializer.save(graph, self.paths.aspect_to_aspect_graph)
        return graph

    def add_sentiments_and_weights_to_nodes(
        self, graph, discourse_trees_df: pd.DataFrame
    ):
        graph, aspect_sentiments = extend_graph_nodes_with_sentiments_and_weights(
            graph, discourse_trees_df
        )
        serializer.save(graph, self.paths.aspect_to_aspect_graph)
        serializer.save(aspect_sentiments, self.paths.aspect_sentiments)
        return graph, aspect_sentiments

    def generate_aht(
        self,
        aht_graph_creation_fn: Callable,
        filter_relation_fn: Callable = None,
        aht_graph_weight_name: str = "pagerank",
        metric_for_aspect_with_max_weight="pagerank",
        aspects_to_skip=None,
        with_aspect_filtering: bool = False,
    ):
        logging.info(f"Experiments for:  {self.paths.experiment_path}")

        discourse_trees_df = (
            self.extract_discourse_trees()
            .pipe(self.extract_discourse_trees_ids_only)
            .pipe(self.extract_sentiment)
            .pipe(self.extract_aspects)
            .pipe(self.extract_edu_rhetorical_rules)
        )

        if with_aspect_filtering:
            discourse_trees_df = self.filter_rules_based_on_aspects_freq(
                discourse_trees_df
            )

        mlflow.log_metric("discourse_tree_df_len", len(discourse_trees_df))

        graph = self.build_aspect_to_aspect_graph(
            discourse_trees_df, filter_relation_fn, aspects_to_skip
        )
        mlflow.log_metric("aspect_2_aspect_graph_edges", graph.number_of_edges())
        mlflow.log_metric("aspect_2_aspect_graph_nodes", graph.number_of_nodes())
        graph, _ = self.add_sentiments_and_weights_to_nodes(graph, discourse_trees_df)
        aht_graph = aht_graph_creation_fn(
            graph,
            max_number_of_nodes=self.aht_max_number_of_nodes,
            weight=aht_graph_weight_name,
        )

        serializer.save(aht_graph, self.paths.aspect_hierarchical_tree)

        aspect_with_max_weight = sort_networkx_attributes(
            nx.get_node_attributes(aht_graph, metric_for_aspect_with_max_weight)
        )[0]

        aht_graph_directed = nx.bfs_tree(
            aht_graph, aspect_with_max_weight, reverse=True
        )
        draw_tree(
            aht_graph_directed,
            self.paths.experiment_path / f"aht_for_{self.paths.experiment_name}",
        )

    def filter_rules_based_on_aspects_freq(
        self, discourse_trees_df: pd.DataFrame
    ) -> pd.DataFrame:
        # generate aspects counter for aspect when in discourse tree appear at least two aspects -
        # the relation could be derived from it
        aspect_counter = Counter(
            flatten(
                flatten(
                    (
                        discourse_trees_df[
                            discourse_trees_df.aspects.apply(
                                lambda aspects: list(flatten(aspects))
                            ).apply(lambda aspects: len(set(aspects)) > 1)
                        ]["aspects"]
                    )
                )
            )
        )
        aspect_counter_df = pd.DataFrame(
            aspect_counter.most_common(), columns=["aspect", "aspect_occurrences"]
        ).sort_values(by="aspect_occurrences", ascending=False)
        aspects_to_filter = aspect_counter_df[
            aspect_counter_df.aspect_occurrences >= self.min_freq_of_aspects
        ].aspect.tolist()

        def filter_aspects(all_aspects, aspects_to_filter_):
            return [
                [
                    aspect.replace("&", "")
                    .replace("/", "")
                    .replace("\\", "")
                    .replace('"', "")
                    for aspect in aspects
                    if aspect in aspects_to_filter_
                ]
                for aspects in all_aspects
            ]

        discourse_trees_df["aspects"] = discourse_trees_df.aspects.apply(
            lambda aspects: filter_aspects(aspects, aspects_to_filter)
        )
        return discourse_trees_df

    def gerani_pipeline(self):
        self.generate_aht(
            aht_graph_creation_fn=gerani_paper_arrg_to_aht,
            filter_relation_fn=rule_filters.filter_rules_gerani,
            aht_graph_weight_name="moi",
            metric_for_aspect_with_max_weight="moi",
            aspects_to_skip=ASPECTS_TO_SKIP,
            with_aspect_filtering=False,
        )

    def our_pipeline(self, use_aspect_clustering: bool = False):
        self.generate_aht(
            aht_graph_creation_fn=partial(
                our_paper_arrg_to_aht, use_aspect_clustering=use_aspect_clustering
            ),
            filter_relation_fn=None,
            aht_graph_weight_name="weight",
            metric_for_aspect_with_max_weight="weight",
            aspects_to_skip=ASPECTS_TO_SKIP,
            with_aspect_filtering=False,
        )

    def our_pipeline_top_n_rules_per_discourse_tree(
        self, use_aspect_clustering: bool = False, top_n: int = 1
    ):
        self.generate_aht(
            aht_graph_creation_fn=partial(
                our_paper_arrg_to_aht, use_aspect_clustering=use_aspect_clustering
            ),
            filter_relation_fn=partial(rule_filters.filter_top_n_rules, top_n=top_n),
            aht_graph_weight_name="weight",
            metric_for_aspect_with_max_weight="weight",
            aspects_to_skip=ASPECTS_TO_SKIP,
            with_aspect_filtering=False,
        )
