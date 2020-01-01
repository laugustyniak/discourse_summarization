import argparse
import json
import logging
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from json.decoder import JSONDecodeError
from os.path import basename, exists, join, split, splitext
from pathlib import Path
from typing import Callable, Sequence, List, Union, Tuple

import nltk
import pandas as pd
from tqdm import tqdm

from aspects.analysis.gerani_graph_analysis import get_dir_moi_for_node
from aspects.aspects.aspect_extractor import AspectExtractor
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.data_io.serializer import Serializer
from aspects.rst.edu_tree_mapper import EDUTreeMapper
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.rst.rst_parser_client import RSTParserClient
from aspects.sentiment.sentiment_client import BiLSTMModel
from aspects.utilities import settings, pandas_utils
from aspects.utilities.data_paths import IOPaths

if not Path('logs').exists():
    Path('logs').mkdir(parents=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s;%(filename)s:%(lineno)s;'
           '%(funcName)s();%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='logs/run.log',
    filemode='w',
)


def extract_discourse_tree(document: str) -> Union[nltk.Tree, None]:
    parser = RSTParserClient()
    try:
        return nltk.Tree.fromstring(
            parser.parse(document),
            leaf_pattern=settings.DISCOURSE_TREE_LEAF_PATTERN
        )
    except (ValueError, JSONDecodeError):
        logging.info(f'Document with errors: {document}')
        return None


def extract_discourse_tree_with_ids_only(discourse_tree: nltk.Tree) -> Tuple[nltk.Tree, List[str]]:
    edu_tree_preprocessor = EDUTreeMapper()
    edu_tree_preprocessor.process_tree(discourse_tree)
    return discourse_tree, edu_tree_preprocessor.edus


def extract_rules(discourse_tree: nltk.Tree) -> List:
    rules_extractor = EDUTreeRulesExtractor(tree=discourse_tree)
    return rules_extractor.extract()


class AspectAnalysisSystem:

    def __init__(
            self,
            input_path: Union[str, Path],
            output_path: Union[str, Path],
            analysis_results_path: Union[str, Path],
            jobs: int = None,
            sent_model_path=None,
            n_logger=1000,
            batch_size=None,
            max_docs=None,
            cycle_in_relations=True,
            filter_gerani=False,
            aht_gerani=False,
            neutral_sent=False
    ):
        self.neutral_sent = neutral_sent
        self.aht_gerani = aht_gerani
        self.filter_gerani = filter_gerani
        self.max_docs = max_docs
        self.batch_size = batch_size
        self.analysis_results_path = analysis_results_path
        self.input_file_path = input_path
        if self.max_docs is not None:
            self.output_path = '{}-{}-docs'.format(output_path, self.max_docs)
        else:
            self.output_path = output_path
        if filter_gerani:
            self.paths = IOPaths(
                input_path, self.output_path, suffix='gerani_one_rule_per_document')
        else:
            self.paths = IOPaths(input_path, self.output_path)
        self.sent_model_path = sent_model_path
        self.serializer = Serializer()
        self.cycle_in_relations = cycle_in_relations

        # number of all processes
        if jobs is None:
            self.jobs = multiprocessing.cpu_count()
        else:
            self.jobs = jobs

        # by how many examples logging will be done
        self.n_logger = n_logger

    def parallelized_extraction(
            self,
            elements: Sequence,
            fn: Callable,
            desc: str = 'Running in parallel'
    ) -> List:
        with ProcessPoolExecutor(self.jobs) as pool:
            return list(
                tqdm(
                    pool.map(
                        fn,
                        elements,
                        chunksize=settings.PARALLEL_CHUNK_SIZE
                    ),
                    total=len(elements),
                    desc=desc
                )
            )

    def extract_discourse_trees(self) -> pd.DataFrame:
        if self.paths.discourse_trees_df.exists():
            return pd.read_pickle(self.paths.discourse_trees_df)
        else:
            print(f'{self.input_file_path} will be loaded!')
            f_extension = basename(self.input_file_path).split('.')[-1]

            if f_extension in ['json']:
                with open(self.input_file_path, 'r') as json_file:
                    df = pd.DataFrame(
                        json.load(json_file).values(), columns=['text'])
            elif f_extension in ['csv', 'txt']:
                df = pd.read_csv(self.input_file_path, header=None)
                df.columns = ['text']
            else:
                raise Exception(
                    'Wrong file type! It must be [json, txt or csv]')

            if self.max_docs is not None:
                df = df.sample(self.max_docs)

            df['discourse_tree'] = self.parallelized_extraction(
                df.text, extract_discourse_tree, 'Discourse trees parsing')

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
            df.discourse_tree, extract_discourse_tree_with_ids_only, 'Discourse trees parsing to idx only')))
        self.discourse_trees_df_checkpoint(df)

        return df

    def discourse_trees_df_checkpoint(self, df):
        self.paths.discourse_trees_df.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(self.paths.discourse_trees_df)

    def extract_sentiment_from_edus(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'sentiment' in df.columns:
            logging.info('Sentiments have been already extracted. Passing to the next step.')
            return df

        pandas_utils.assert_columns(df, 'edus')
        analyzer = BiLSTMModel()
        df['sentiment'] = self.parallelized_extraction(df.edus, analyzer.get_sentiments, 'Sentiment extracting')
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_aspects_from_edus(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'aspects' in df.columns:
            logging.info('Aspects have been already extracted. Passing to the next step.')
            return df

        pandas_utils.assert_columns(df, 'edus')

        extractor = AspectExtractor()
        df['aspects'] = self.parallelized_extraction(df.edus, extractor.extract_batch, 'Aspects extracting')
        self.discourse_trees_df_checkpoint(df)

        df['concepts'] = self.parallelized_extraction(
            df.aspects, extractor.extract_concepts_batch, 'Concepts extracting')
        self.discourse_trees_df_checkpoint(df)

        return df

    def extract_edu_dependency_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: ended here, ids of DT idx only in the beggining of tree coulkd be 0 not a Tree instance, same in the end of tree
        if 'rules' in df.columns:
            return df

        pandas_utils.assert_columns(df, 'discourse_tree_ids_only')

        df['rules'] = self.parallelized_extraction(df.discourse_tree_ids_only, extract_rules, 'Extracting rules')

        self.discourse_trees_df_checkpoint(df)

        return df

    def _build_aspect_dependency_graph(self):
        """Build dependency graph"""

        if not (exists(self.paths.aspects_graph) and exists(self.paths.aspects_page_ranks)):
            dependency_rules = self.serializer.load(
                self.paths.edu_dependency_rules)
            aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
            documents_info = self.serializer.load(self.paths.docs_info)

            builder = AspectsGraphBuilder(
                aspects_per_edu, with_cycles_between_aspects=self.cycle_in_relations)
            graph, page_ranks = builder.build(
                rules=dependency_rules,
                docs_info=documents_info,
                conceptnet_io=settings.CONCEPTNET_IO_ASPECTS,
                filter_gerani=self.filter_gerani,
                aht_gerani=self.aht_gerani,
                aspect_graph_path=self.paths.aspects_graph,
            )

            self.serializer.save(graph, self.paths.aspects_graph)
            self.serializer.save(page_ranks, self.paths.aspects_page_ranks)

    def _filter_aspects(self, threshold):
        """Filter out aspects according to threshold"""
        aspects_importance = self.serializer.load(
            self.paths.aspects_page_ranks)
        documents_info = self.serializer.load(self.paths.docs_info)

        aspects_count = len(aspects_importance)
        aspects_list = list(aspects_importance)

        for documentId, document_info in tqdm(
                documents_info.iteritems(), desc='Filter aspects', total=len(documents_info)):
            aspects = []
            if 'aspects' in document_info:
                for aspect in document_info['aspects']:
                    if aspect in aspects_importance:
                        aspect_position = float(
                            aspects_list.index(aspect) + 1) / aspects_count
                        if aspect_position < threshold:
                            aspects.append(aspect)
            documents_info[documentId]['aspects'] = aspects
        self.serializer.save(documents_info, self.paths.final_docs_info)

    def _add_sentiment_and_dir_moi_to_graph(self):
        aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
        documents_info = self.serializer.load(self.paths.docs_info)
        aspect_graph = self.serializer.load(self.paths.aspects_graph)
        aspect_graph = get_dir_moi_for_node(
            aspect_graph, aspects_per_edu, documents_info)
        self.serializer.save(aspect_graph, self.paths.aspects_graph)

    def run(self):

        discourse_trees_df = self.extract_discourse_trees()
        discourse_trees_df = self.extract_discourse_trees_ids_only(discourse_trees_df)
        discourse_trees_df = self.extract_sentiment_from_edus(discourse_trees_df)
        discourse_trees_df = self.extract_aspects_from_edus(discourse_trees_df)

        discourse_trees_df = self.extract_edu_dependency_rules(discourse_trees_df)
        # self._build_aspect_dependency_graph()
        # self._add_sentiment_and_dir_moi_to_graph()
        # aspects_graph = self.serializer.load(self.paths.aspects_graph)
        # nx.write_gpickle(aspects_graph, self.paths.aspects_graph_gpkl)
        # # for gephi
        # nx.write_gexf(aspects_graph, self.paths.aspects_graph_gexf)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Process documents.')
    arg_parser.add_argument(
        '-input',
        type=str,
        dest='input_file_path',
        default=settings.DEFAULT_INPUT_FILE_PATH,
        help='Path to the file with documents (json, csv, pickle)'
    )
    arg_parser.add_argument(
        '-output',
        type=str,
        dest='output_file_path',
        default=settings.DEFAULT_OUTPUT_PATH,
        help='Number of processes'
    )
    arg_parser.add_argument(
        '-sent_model',
        type=str,
        dest='sent_model_path',
        default=None,
        help='path to sentiment model'
    )
    arg_parser.add_argument(
        '-analysis_results_path',
        type=str,
        dest='analysis_results_path',
        default=None,
        help='path to analysis results'
    )
    arg_parser.add_argument(
        '-max_docs',
        type=int,
        dest='max_docs',
        default=None,
        help='Maximum number of documents to analyse'
    )
    arg_parser.add_argument(
        '-batch', type=int, dest='batch_size', default=None, help='batch size for each process')
    arg_parser.add_argument(
        '-p', type=int, dest='max_processes', default=1, help='Number of processes')
    arg_parser.add_argument(
        '-cycles',
        type=bool,
        dest='cycles',
        default=False,
        help='Do we want to have cycles in aspect relation? False by default'
    )
    arg_parser.add_argument(
        '-filter_gerani',
        type=bool,
        dest='filter_gerani',
        default=False,
        help='Do we want to follow Gerani paper?'
    )
    arg_parser.add_argument(
        '-aht_gerani',
        type=bool,
        dest='aht_gerani',
        default=False,
        help='Do we want to create AHT by Gerani?'
    )
    arg_parser.add_argument(
        '-neutral_sent',
        type=bool,
        dest='neutral_sent',
        default=False,
        help='Do we want to use neutral sentiment aspects too?'
    )
    args = arg_parser.parse_args()

    input_file_full_name = split(args.input_file_path)[1]
    input_file_name = splitext(input_file_full_name)[0]
    output_path = join(args.output_file_path, input_file_name)

    AAS = AspectAnalysisSystem(
        input_path=args.input_file_path,
        output_path=output_path,
        analysis_results_path=args.analysis_results_path,
        jobs=args.max_processes,
        sent_model_path=args.sent_model_path,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        cycle_in_relations=True if args.cycles else False,
        filter_gerani=args.filter_gerani,
        aht_gerani=args.aht_gerani,
        neutral_sent=args.neutral_sent
    )
    AAS.run()
