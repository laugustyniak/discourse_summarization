import argparse
import json
import logging
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from os.path import basename, dirname, exists, join, split, splitext
from pathlib import Path

import networkx as nx
import nltk
import pandas as pd
from tqdm import tqdm

from aspects.analysis.gerani_graph_analysis import get_dir_moi_for_node
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.aspects.edu_aspect_extractor import EDUAspectExtractor
from aspects.data_io.serializer import Serializer
from aspects.rst.edu_tree_preprocesser import EDUTreePreprocessor
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.rst.rst_parser_client import RSTParserClient
from aspects.sentiment.sentiment_analyzer import (
    LogisticRegressionSentimentAnalyzer as SentimentAnalyzer
)
from aspects.utilities import settings
from aspects.utilities.data_paths import IOPaths
from aspects.utilities.settings import DISCOURSE_TREE_LEAF_PATTERN

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

PARALLEL_CHUNK_SIZE = 100

def extract_discourse_tree(document):
    parser = RSTParserClient()
    return nltk.Tree.fromstring(parser.parse(document), leaf_pattern=DISCOURSE_TREE_LEAF_PATTERN)

    
def extract_discourse_tree_with_ids_only(discourse_tree):
    edu_tree_preprocessor = EDUTreePreprocessor()
    # TODO: rewrite it with simple return fashion not change the state of the tree
    edu_tree_preprocessor.process_tree(discourse_tree)
    return discourse_tree, edu_tree_preprocessor.get_preprocessed_edus()

class AspectAnalysisSystem:

    def __init__(
            self,
            input_path,
            output_path,
            gold_standard_path,
            analysis_results_path,
            jobs=1,
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
        self.gold_standard_path = gold_standard_path
        self.analysis_results_path = analysis_results_path
        self.input_file_path = input_path
        if self.max_docs is not None:
            self.output_path = '{}-{}-docs'.format(output_path, self.max_docs)
        else:
            self.output_path = output_path
        if filter_gerani:
            self.paths = IOPaths(input_path, self.output_path, suffix='gerani_one_rule_per_document')
        else:
            self.paths = IOPaths(input_path, self.output_path)
        self.sent_model_path = sent_model_path
        self.serializer = Serializer()
        self.cycle_in_relations = cycle_in_relations

        # number of all processes
        self.jobs = jobs

        # by how many examples logging will be done
        self.n_logger = n_logger

        # count number of error within parsing RDT
        self.parsing_errors = 0

    def extract_discourse_trees(self):
        discourse_tree_df_path = Path(self.paths.discourse_trees_df)

        if discourse_tree_df_path.exists():
            return pd.read_pickle(discourse_tree_df_path)
        else:
            f_extension = basename(self.input_file_path).split('.')[-1]

            if f_extension in ['json']:
                df = pd.DataFrame(json.load(open(self.input_file_path, 'r')).values(), columns=['text'])
            elif f_extension in ['csv', 'txt']:
                df = pd.read_csv(self.input_file_path, header=None)
                df.columns = ['text']
            else:
                raise Exception('Wrong file type! It must be [json, txt or csv]')

            if self.max_docs is not None:
                df = df.head(self.max_docs)
            
            # Process the rows in chunks in parallel
            with ProcessPoolExecutor(self.jobs) as pool:
                df['discourse_tree'] = list(
                    tqdm(pool.map(
                        extract_discourse_tree, 
                        df['text'], 
                        chunksize=PARALLEL_CHUNK_SIZE
                    ), total=df.shape[0], desc='Discourse trees parsing')
                )
                df['discourse_tree_ids_only'], df['edus'] = tuple(zip(*list(
                    tqdm(pool.map(
                        extract_discourse_tree_with_ids_only, 
                        df['discourse_tree'], 
                        chunksize=PARALLEL_CHUNK_SIZE
                    ), total=df.shape[0], desc='Discourse trees parsing to idx only')
                )))

            discourse_tree_df_path.parent.mkdir(parents=True)
            df.to_pickle(discourse_tree_df_path)
        
        return df

    def _filter_edu_by_sentiment(self):
        """Filter out EDUs without sentiment, with neutral sentiment too"""

        if not (exists(self.paths.sentiment_filtered_edus)
                and exists(self.paths.docs_info)):

            if self.sent_model_path is None:
                analyzer = SentimentAnalyzer()
            else:
                analyzer = SentimentAnalyzer(model_path=self.sent_model_path)
            edus = self.serializer.load(self.paths.raw_edus)
            filtered_edus = {}
            docs_info = {}

            for edu_id, edu in tqdm(edus.items(), desc='Filtering based on Sentiment Analysis of EDUs'):
                edu['sentiment'] = []
                logging.debug('edu: {}'.format(edu))
                sentiment = analyzer.analyze(edu['text'])[0]

                # todo add to readme structure of document info and other dicts
                if not edu['source_document_id'] in docs_info:
                    docs_info[edu['source_document_id']] = {
                        'EDUs': [],
                        'accepted_edus': [],
                        'aspects': {},
                        'aspect_concepts': {},
                        'aspect_keywords': {},
                        'sentiment': {},
                    }
                docs_info[edu['source_document_id']]['sentiment'].update({edu_id: sentiment})
                docs_info[edu['source_document_id']]['EDUs'].append(edu_id)
                if sentiment or self.neutral_sent:
                    # FIXME: why list not float/int as sentiment?
                    edu['sentiment'].append(sentiment)
                    docs_info[edu['source_document_id']]['accepted_edus'].append(edu_id)
                    filtered_edus[edu_id] = edu
            self.serializer.save(filtered_edus, self.paths.sentiment_filtered_edus)
            self.serializer.save(docs_info, self.paths.docs_info)

    def _extract_aspects_from_edu(self):
        """ Extract aspects from EDU and serialize them """
        if exists(self.paths.aspects_per_edu):
            aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
            logging.info('Aspect per EDU loaded.')
        else:
            logging.info('No aspects extracted, starting from beginning!')
            aspects_per_edu = {}

        extractor = EDUAspectExtractor()
        edus = self.serializer.load(self.paths.sentiment_filtered_edus)
        documents_info = self.serializer.load(self.paths.docs_info)
        n_edus = len(edus)

        logging.info('# of document with sentiment edus: {}'.format(n_edus))

        for edu_id, edu in tqdm(edus.items(), desc='Aspect Extraction from EDUs'):
            if edu_id not in aspects_per_edu:
                doc_info = documents_info[edu['source_document_id']]
                aspects, aspect_concepts, aspect_keywords = extractor.extract(edu)
                aspects_per_edu[edu_id] = aspects
                logging.info('EDU ID/MAX EDU ID: {}'.format(edu_id))
                logging.debug('aspects: {}'.format(aspects))
                if 'aspects' not in doc_info:
                    doc_info['aspects'] = []
                doc_info['aspects'].update({edu_id: aspects})
                doc_info['aspect_concepts'].update({edu_id: aspect_concepts})
                doc_info['aspect_keywords'].update({edu_id: aspect_keywords})

                if not edu_id % settings.ASPECT_EXTRACTION_SERIALIZATION_STEP:
                    logging.info('Save partial aspects, edu_id {}'.format(edu_id))
                    self.serializer.save(aspects_per_edu, self.paths.aspects_per_edu)
                    self.serializer.save(documents_info, self.paths.docs_info)

        logging.info('Serializing aspect per edu and document info objects.')
        self.serializer.save(aspects_per_edu, self.paths.aspects_per_edu)
        self.serializer.save(documents_info, self.paths.docs_info)

    def _extract_edu_dependency_rules(self):
        """Extract association rules to RST trees"""
        if not exists(self.paths.edu_dependency_rules):
            link_tree = None
            rules_extractor = EDUTreeRulesExtractor()
            rules = {}
            docs_info = self.serializer.load(self.paths.docs_info)
            for doc_id, doc_info in tqdm(
                    docs_info.iteritems(), desc='Extract EDU dependency rules', total=len(docs_info)):
                if len(doc_info['accepted_edus']) > 0:
                    link_tree = self.serializer.load(join(self.paths.discourse_trees_df, str(doc_id)))
                extracted_rules = rules_extractor.extract(link_tree, doc_info['accepted_edus'], doc_id)
                rules.update(extracted_rules)
            logging.info('Rules extracted.')
            self.serializer.save(rules, self.paths.edu_dependency_rules)

    def _build_aspect_dependency_graph(self):
        """Build dependency graph"""

        if not (exists(self.paths.aspects_graph) and exists(self.paths.aspects_page_ranks)):
            dependency_rules = self.serializer.load(self.paths.edu_dependency_rules)
            aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
            documents_info = self.serializer.load(self.paths.docs_info)

            builder = AspectsGraphBuilder(aspects_per_edu, with_cycles_between_aspects=self.cycle_in_relations)
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
        aspects_importance = self.serializer.load(self.paths.aspects_page_ranks)
        documents_info = self.serializer.load(self.paths.docs_info)

        aspects_count = len(aspects_importance)
        aspects_list = list(aspects_importance)

        for documentId, document_info in tqdm(
                documents_info.iteritems(), desc='Filter aspects', total=len(documents_info)):
            aspects = []
            if 'aspects' in document_info:
                for aspect in document_info['aspects']:
                    if aspect in aspects_importance:
                        aspect_position = float(aspects_list.index(aspect) + 1) / aspects_count
                        if aspect_position < threshold:
                            aspects.append(aspect)
            documents_info[documentId]['aspects'] = aspects
        self.serializer.save(documents_info, self.paths.final_docs_info)

    def _add_sentiment_and_dir_moi_to_graph(self):
        aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
        documents_info = self.serializer.load(self.paths.docs_info)
        aspect_graph = self.serializer.load(self.paths.aspects_graph)
        aspect_graph = get_dir_moi_for_node(aspect_graph, aspects_per_edu, documents_info)
        self.serializer.save(aspect_graph, self.paths.aspects_graph)

    def run(self):

        discourse_trees_df = self.extract_discourse_trees()

        self._filter_edu_by_sentiment()
        self._extract_aspects_from_edu()
        self._extract_edu_dependency_rules()
        self._build_aspect_dependency_graph()
        self._add_sentiment_and_dir_moi_to_graph()
        aspects_graph = self.serializer.load(self.paths.aspects_graph)
        nx.write_gpickle(aspects_graph, self.paths.aspects_graph_gpkl)
        # for gehpi
        nx.write_gexf(aspects_graph, self.paths.aspects_graph_gexf)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Process documents.')
    arg_parser.add_argument('-input', type=str, dest='input_file_path', default=settings.DEFAULT_INPUT_FILE_PATH,
                            help='Path to the file with documents (json, csv, pickle)')
    arg_parser.add_argument('-output', type=str, dest='output_file_path', default=settings.DEFAULT_OUTPUT_PATH,
                            help='Number of processes')
    arg_parser.add_argument('-sent_model', type=str, dest='sent_model_path', default=None,
                            help='path to sentiment model')
    arg_parser.add_argument('-analysis_results_path', type=str, dest='analysis_results_path', default=None,
                            help='path to analysis results')
    arg_parser.add_argument('-max_docs', type=int, dest='max_docs', default=None,
                            help='Maximum number of documents to analyse')
    arg_parser.add_argument('-batch', type=int, dest='batch_size', default=None,
                            help='batch size for each process')
    arg_parser.add_argument('-p', type=int, dest='max_processes', default=1,
                            help='Number of processes')
    arg_parser.add_argument('-cycles', type=bool, dest='cycles', default=False,
                            help='Do we want to have cycles in aspect realation? False by default')
    arg_parser.add_argument('-filter_gerani', type=bool, dest='filter_gerani', default=False,
                            help='Do we want to follow Gerani paper?')
    arg_parser.add_argument('-aht_gerani', type=bool, dest='aht_gerani', default=False,
                            help='Do we want to create AHT by Gerani?')
    arg_parser.add_argument('-neutral_sent', type=bool, dest='neutral_sent', default=False,
                            help='Do we want to use neutral sentiment aspects too?')
    args = arg_parser.parse_args()

    input_file_full_name = split(args.input_file_path)[1]
    input_file_name = splitext(input_file_full_name)[0]
    output_path = join(args.output_file_path, input_file_name)
    gold_standard_path = dirname(args.input_file_path) + input_file_name + '_aspects_list.ser'
    AAS = AspectAnalysisSystem(
        input_path=args.input_file_path,
        output_path=output_path,
        gold_standard_path=gold_standard_path,
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
