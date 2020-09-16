import json
import logging
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from os.path import basename
from pathlib import Path
from typing import Callable, Sequence, List, Union

import mlflow
import networkx as nx
import pandas as pd
from tqdm import tqdm

from aspects.analysis.gerani_graph_analysis import (
    extend_graph_nodes_with_sentiments_and_weights,
    gerani_paper_arrg_to_aht,
    our_paper_arrg_to_aht)
from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.aspects.aspects_graph_builder import Aspect2AspectGraph, sort_networkx_attibutes
from aspects.aspects.rule_filters import filter_rules_gerani
from aspects.data_io import serializer
from aspects.rst.extractors import extract_discourse_tree, extract_discourse_tree_with_ids_only, extract_rules
from aspects.sentiment.simple_textblob import analyze
from aspects.utilities import settings, pandas_utils
from aspects.utilities.data_paths import ExperimentPaths
from aspects.visualization.drawing import draw_tree

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s;%(filename)s:%(lineno)s;'
           '%(funcName)s();%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='analyzer.log'
)


class AspectAnalysis:

    def __init__(
            self,
            input_path: Union[str, Path],
            output_path: Union[str, Path],
            experiment_name: str = '',
            jobs: int = None,
            sent_model_path: Union[str, Path] = None,
            batch_size: int = None,
            max_docs: int = None,
            alpha_coefficient: float = 0.5,
            aht_max_number_of_nodes: int = 50,
    ):
        self.max_docs = max_docs
        mlflow.log_param('max_docs', max_docs)
        self.batch_size = batch_size
        self.input_file_path = input_path
        if self.max_docs is not None:
            self.output_path = f'{str(output_path)}-{self.max_docs}-docs'
        else:
            self.output_path = output_path
        self.experiment_name = experiment_name
        self.paths = ExperimentPaths(input_path, self.output_path, experiment_name)
        self.sent_model_path = sent_model_path
        mlflow.log_param('sent_model_path', sent_model_path)

        # number of all processes
        if jobs is None:
            self.jobs = multiprocessing.cpu_count()
        else:
            self.jobs = jobs

        self.alpha_coefficient = alpha_coefficient
        mlflow.log_param('alpha_coefficient', alpha_coefficient)
        self.aht_max_number_of_nodes = aht_max_number_of_nodes
        mlflow.log_param('aht_max_number_of_nodes', aht_max_number_of_nodes)

    def parallelized_extraction(
            self,
            elements: Sequence,
            fn: Callable,
            desc: str = 'Running in parallel'
    ) -> List:
        with ProcessPoolExecutor(self.jobs) as pool:
            return list(
                tqdm(
                    pool.map(fn, elements, chunksize=self.batch_size),
                    total=len(elements),
                    desc=desc
                )
            )

    def extract_discourse_trees(self) -> pd.DataFrame:
        if self.paths.discourse_trees_df.exists():
            logging.info('Discourse trees loading.')
            return pd.read_pickle(self.paths.discourse_trees_df)
        else:
            print(f'{self.input_file_path} will be loaded!')
            f_extension = basename(self.input_file_path).split('.')[-1]

            if f_extension in ['json']:
                with open(self.input_file_path, 'r') as json_file:
                    df = pd.DataFrame(json.load(json_file).values(), columns=['text'])
            elif f_extension in ['csv', 'txt']:
                df = pd.read_csv(self.input_file_path, header=None)
                df.columns = ['text']
            else:
                raise Exception('Wrong file type! It must be [json, txt or csv]')

            if self.max_docs is not None:
                df = df.head(self.max_docs)

            df['discourse_tree'] = self.parallelized_extraction(
                df.text.tolist(), extract_discourse_tree, 'Discourse trees parsing')

            n_docs = len(df)
            df.dropna(subset=['discourse_tree'], inplace=True)
            logging.info(f'{n_docs - len(df)} discourse tree has been parser with errors and we skip them.')

            assert not df.empty, 'No trees to process!'

            self.discourse_trees_df_checkpoint(df)

        return df

    def extract_discourse_trees_ids_only(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'discourse_tree_ids_only' in df.columns:
            return df

        df['discourse_tree_ids_only'], df['edus'] = tuple(zip(*self.parallelized_extraction(
            df.discourse_tree.tolist(),
            extract_discourse_tree_with_ids_only,
            'Discourse trees parsing to idx only'
        )))
        self.discourse_trees_df_checkpoint(df)

        return df

    def discourse_trees_df_checkpoint(self, df: pd.DataFrame):
        logging.info(f'Discourse data frame - saving.')
        df.to_pickle(self.paths.discourse_trees_df)
        logging.info(f'Discourse data frame - saved.')

    def extract_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sentiment' in df.columns:
            logging.info('Sentiments have been already extracted. Passing to the next step.')
            return df

        pandas_utils.assert_columns(df, 'edus')
        df['sentiment'] = self.parallelized_extraction(
            df.edus.tolist(),
            analyze,
            'Sentiment extracting'
        )
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'aspects' in df.columns:
            logging.info('Aspects have been already extracted. Passing to the next step.')
            return df

        pandas_utils.assert_columns(df, 'edus')

        extractor = AspectExtractor()
        df['aspects'] = self.parallelized_extraction(df.edus.tolist(), extractor.extract_batch, 'Aspects extracting')
        self.discourse_trees_df_checkpoint(df)

        df['concepts'] = self.parallelized_extraction(
            df.aspects.tolist(), extractor.extract_concepts_batch, 'Concepts extracting')
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_edu_rhetorical_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'rules' in df.columns:
            return df

        pandas_utils.assert_columns(df, 'discourse_tree_ids_only')

        df['rules'] = self.parallelized_extraction(
            df.discourse_tree_ids_only.tolist(), extract_rules, 'Extracting rules')

        self.discourse_trees_df_checkpoint(df)

        return df

    def build_aspect_to_aspect_graph(self, df: pd.DataFrame, filter_relation_fn: Callable = None):
        if self.paths.aspect_to_aspect_graph.exists():
            logging.info('Aspect to aspect graph has been already built, loading and passing to the next stage')
            return serializer.load(self.paths.aspect_to_aspect_graph)

        builder = Aspect2AspectGraph()
        graph = builder.build(
            discourse_tree_df=df,
            conceptnet_io=settings.CONCEPTNET_IO_ASPECTS,
            filter_relation_fn=filter_relation_fn
        )
        serializer.save(graph, self.paths.aspect_to_aspect_graph)
        return graph

    def add_sentiments_and_weights_to_nodes(self, graph, discourse_trees_df: pd.DataFrame):
        graph, aspect_sentiments = extend_graph_nodes_with_sentiments_and_weights(graph, discourse_trees_df)
        serializer.save(graph, self.paths.aspect_to_aspect_graph)
        serializer.save(aspect_sentiments, self.paths.aspect_sentiments)
        return graph, aspect_sentiments

    def gerani_pipeline(self):
        logging.info(f'Experiments for:  {self.paths.experiment_path}')

        discourse_trees_df = (self.extract_discourse_trees()
                              .pipe(self.extract_discourse_trees_ids_only)
                              .pipe(self.extract_sentiment)
                              .pipe(self.extract_aspects)
                              .pipe(self.extract_edu_rhetorical_rules)
                              )

        mlflow.log_metric('discourse_tree_df_len', len(discourse_trees_df))
        graph = self.build_aspect_to_aspect_graph(discourse_trees_df, filter_rules_gerani)
        mlflow.log_metric('asspect_2_aspect_graph_edges', graph.number_of_edges())
        mlflow.log_metric('asspect_2_aspect_graph_nodes', graph.number_of_nodes())
        graph, aspect_sentiments = self.add_sentiments_and_weights_to_nodes(graph, discourse_trees_df)
        aht_graph = gerani_paper_arrg_to_aht(graph, max_number_of_nodes=self.aht_max_number_of_nodes, weight='moi')

        serializer.save(aht_graph, self.paths.aspect_hierarchical_tree)

        aspect_with_max_pagerank = sort_networkx_attibutes(nx.get_node_attributes(aht_graph, 'pagerank'))[0]
        aht_graph_directed = nx.bfs_tree(aht_graph, aspect_with_max_pagerank)
        mlflow.log_param('aspect_with_max_pagerank', aspect_with_max_pagerank)
        # draw_tree(aht_graph_directed, self.paths.experiment_path / 'aht_gerani_based_root_node_highest_wpr')

    def our_pipeline(self):
        logging.info(f'Experiments for:  {self.paths.experiment_path}')

        discourse_trees_df = (self.extract_discourse_trees()
                              .pipe(self.extract_discourse_trees_ids_only)
                              .pipe(self.extract_sentiment)
                              .pipe(self.extract_aspects)
                              .pipe(self.extract_edu_rhetorical_rules)
                              )

        mlflow.log_metric('discourse_tree_df_len', len(discourse_trees_df))
        graph = self.build_aspect_to_aspect_graph(discourse_trees_df)
        mlflow.log_metric('asspect_2_aspect_graph_nodes', graph.number_of_nodes())
        mlflow.log_metric('asspect_2_aspect_graph_edges', graph.number_of_edges())
        graph, aspect_sentiments = self.add_sentiments_and_weights_to_nodes(graph, discourse_trees_df)
        aht_graph = our_paper_arrg_to_aht(graph, max_number_of_nodes=self.aht_max_number_of_nodes, weight='pagerank')

        serializer.save(aht_graph, self.paths.aspect_hierarchical_tree)

        aspect_with_max_pagerank = sort_networkx_attibutes(nx.get_node_attributes(aht_graph, 'pagerank'))[0]
        mlflow.log_param('aspect_with_max_pagerank', aspect_with_max_pagerank)
        aht_graph_directed = nx.bfs_tree(aht_graph, aspect_with_max_pagerank)
        # draw_tree(aht_graph_directed, self.paths.experiment_path / 'aht_gerani_based_root_node_highest_wpr')
