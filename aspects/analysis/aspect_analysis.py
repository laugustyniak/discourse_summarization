import argparse
import logging
from os import getcwd
from os.path import basename, exists, join, split, splitext, dirname

import networkx as nx
import pandas as pd
import simplejson
from tqdm import tqdm

from aspects.analysis.gerani_graph_analysis import get_dir_moi_for_node
from aspects.analysis.results_analyzer import ResultsAnalyzer
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.aspects import aspect_extractor
from aspects.configs.conceptnets_config import CONCEPTNET_ASPECTS
from aspects.io.serializer import Serializer
from aspects.rst.edu_tree_preprocesser import EduExtractor
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.sentiment.sentiment_analyzer import load_sentiment_analyzer
from aspects.utilities.data_paths import IOPaths
from parse import DiscourseParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s;%(filename)s:%(lineno)s;'
                           '%(funcName)s();%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='logs/run.log',
                    filemode='w',
                    )


class AspectAnalysis:
    def __init__(self, input_path, output_path, gold_standard_path, analysis_results_path, jobs=1, sent_model_path=None,
                 n_logger=1000, batch_size=None, n_sample=10):

        self.batch_size = batch_size
        self.gold_standard_path = gold_standard_path
        self.analysis_results_path = analysis_results_path
        self.input_file_path = input_path
        self.paths = IOPaths(input_path, output_path)
        self.sent_model_path = sent_model_path
        self.serializer = Serializer()

        # number of all processes
        self.jobs = jobs

        # by how many examples logging will be done
        self.n_loger = n_logger

        # count number of error within parsing RDT
        self.parsing_errors = 0

        self.n_sample = n_sample
        self.edu_preprocesser = EduExtractor()

    def parse_input_documents(self):
        """
        Load and parse documents. All document should be stored in
        JSON/dictionary format, only values will be processed.

        :return:
            documents_count : pd.DataFrame
                Number of documents processed
        """
        if exists(self.paths.extracted_docs_all):
            f_extension = basename(self.input_file_path).split('.')[-1]
            logging.debug('Input file extension: {}'.format(f_extension))
            # TODO: load list of documents or generator
            with open(self.input_file_path, 'r') as f:
                raw_documents = simplejson.load(f)
                docs = [(doc_id, unicode(document)) for doc_id, document in raw_documents.iteritems()][:self.n_sample]
                documents_df = pd.DataFrame(docs, columns=['doc_id', 'document'])
                logging.info('Number of all documents to analyse: {}'.format(documents_df.shape[0]))
                documents_df.to_csv(self.paths.extracted_docs_all)
        else:
            documents_df = pd.read_csv(self.paths.extracted_docs_all, encoding='utf8')
        return documents_df

    def parse_edus(self, documents_df):
        parser = DiscourseParser(output_dir=self.paths.tmp)
        documents_df['rst_tree'] = [parser.parse(doc) for doc in tqdm(documents_df.document, desc='RST parsing')]
        documents_df.to_pickle(self.paths.edu_trees_df)
        return documents_df

    def extract_edus(self, documents_df):
        edus = [self.edu_preprocesser.get_edus_with_idsfrom_tree(tree) for tree in
                tqdm(documents_df['rst_tree'], desc='Edu extraction')]
        documents_df['edus'] = edus
        return documents_df

    def get_sentiment(self, edu_df):
        sentiment_model = load_sentiment_analyzer()
        edu_df['sentiment'] = [sentiment_model.predict([edu.text])[0] for edu in tqdm(edu_df['edu'])]
        return edu_df

    def _extract_edu_dependency_rules(self):
        """Extract association rules to RST trees"""
        if not exists(self.paths.edu_dependency_rules):
            link_tree = None
            rules_extractor = EDUTreeRulesExtractor()
            rules = {}
            docs_info = self.serializer.load(self.paths.docs_info)
            for doc_id, doc_info in docs_info.iteritems():
                if len(doc_info['accepted_edus']) > 0:
                    link_tree = self.serializer.load(join(self.paths.link_trees, str(doc_id)))
                extracted_rules = rules_extractor.extract(link_tree, doc_info['accepted_edus'], doc_id)
                rules.update(extracted_rules)
            logging.info('Rules extracted.')
            self.serializer.save(rules, self.paths.edu_dependency_rules)

    def _build_aspect_dependency_graph(self):
        """Build dependency graph"""

        if not (exists(self.paths.aspects_graph) and exists(self.paths.aspects_importance)):
            dependency_rules = self.serializer.load(self.paths.edu_dependency_rules)
            aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
            documents_info = self.serializer.load(self.paths.docs_info)

            builder = AspectsGraphBuilder(aspects_per_edu)
            graph, page_ranks = builder.build(rules=dependency_rules,
                                              docs_info=documents_info,
                                              conceptnet_io=CONCEPTNET_ASPECTS,
                                              filter_gerani=False,
                                              aht_gerani=False,
                                              aspect_graph_path=self.paths.aspects_graph,
                                              )

            self.serializer.save(graph, self.paths.aspects_graph)
            self.serializer.save(page_ranks, self.paths.aspects_importance)

    def _filter_aspects(self, threshold):
        """Filter out aspects according to threshold"""
        aspects_importance = self.serializer.load(self.paths.aspects_importance)
        documents_info = self.serializer.load(self.paths.docs_info)

        aspects_count = len(aspects_importance)
        aspects_list = list(aspects_importance)

        for documentId, document_info in documents_info.iteritems():
            aspects = []
            if 'aspects' in document_info:
                for aspect in document_info['aspects']:
                    if aspect in aspects_importance:
                        aspect_position = float(aspects_list.index(aspect) + 1) / aspects_count
                        if aspect_position < threshold:
                            aspects.append(aspect)
            documents_info[documentId]['aspects'] = aspects
        self.serializer.save(documents_info, self.paths.final_docs_info)

    def _analyze_results(self, threshold):
        """ remove noninformative aspects  """
        documents_info = self.serializer.load(self.paths.final_docs_info)
        gold_standard = self.serializer.load(self.gold_standard_path)
        if gold_standard is None:
            raise ValueError('GoldStandard data is None')
        analyzer = ResultsAnalyzer()
        for document_id, document_info in documents_info.iteritems():
            analyzer.analyze(document_info['aspects'], gold_standard[document_id])
        measures = analyzer.get_analysis_results()
        self.serializer.append_serialized(
            ';'.join(str(x) for x in [threshold] + measures) + '\n',
            self.analysis_results_path)

    def _add_sentiment_and_dir_moi_to_graph(self):
        aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
        documents_info = self.serializer.load(self.paths.docs_info)
        aspect_graph = self.serializer.load(self.paths.aspects_graph)
        aspect_graph = get_dir_moi_for_node(aspect_graph, aspects_per_edu, documents_info)
        self.serializer.save(aspect_graph, self.paths.aspects_graph)

    def run(self):
        documents_df = (self.parse_input_documents()
                        .pipe(self.parse_edus)
                        .pipe(self.extract_edus)
                        )

        # get edu-based dataframe for sentiment and aspect extraction
        edu_df = pd.DataFrame([(edu_id, edu) for edus in documents_df.edus for edu_id, edu in edus.items()],
                              columns=['edu_id', 'edu'])
        edu_df = (edu_df
                  .pipe(self.get_sentiment)
                  .query('sentiment != 0')
                  .pipe(aspect_extractor.extract_noun_and_noun_phrases)
                  .pipe(aspect_extractor.extract_named_entities)
                  .pipe(aspect_extractor.extract_sentic_concepts)
                  .pipe(aspect_extractor.extract_conceptnet_concepts)
                  .pipe(aspect_extractor.extract_keywords_rake)
                  )

        #  TODO: we are here
        self._extract_edu_dependency_rules()
        self._build_aspect_dependency_graph()
        self._add_sentiment_and_dir_moi_to_graph()
        aspects_graph = self.serializer.load(self.paths.aspects_graph)
        nx.write_gpickle(aspects_graph, self.paths.aspects_graph_gpkl)
        nx.write_gexf(aspects_graph, self.paths.aspects_graph_gexf)


if __name__ == "__main__":
    ROOT_PATH = getcwd()
    DEFAULT_OUTPUT_PATH = join(ROOT_PATH, 'results')
    DEFAULT_INPUT_FILE_PATH = join(ROOT_PATH, 'texts', 'test.txt')

    arg_parser = argparse.ArgumentParser(description='Process documents.')
    arg_parser.add_argument('-input', type=str, dest='input_file_path',
                            default=DEFAULT_INPUT_FILE_PATH,
                            help='Path to the file with documents '
                                 '(json, csv, pickle)')
    arg_parser.add_argument('-output', type=str, dest='output_file_path',
                            default=DEFAULT_OUTPUT_PATH,
                            help='Number of processes')
    arg_parser.add_argument('-sent_model', type=str, dest='sent_model_path',
                            default=None,
                            help='path to sentiment model')
    arg_parser.add_argument('-analysis_results_path', type=str,
                            dest='analysis_results_path',
                            default=None,
                            help='path to analysis results')
    arg_parser.add_argument('-batch', type=int, dest='batch_size', default=None,
                            help='batch size for each process')
    arg_parser.add_argument('-p', type=int, dest='max_processes', default=1,
                            help='Number of processes')
    args = arg_parser.parse_args()

    input_file_full_name = split(args.input_file_path)[1]
    input_file_name = splitext(input_file_full_name)[0]
    output_path = join(args.output_file_path, input_file_name)
    gold_standard_path = dirname(
        args.input_file_path) + input_file_name + '_aspects_list.ser'
    AAS = AspectAnalysis(input_path=args.input_file_path,
                         output_path=output_path,
                         gold_standard_path=gold_standard_path,
                         analysis_results_path=args.analysis_results_path,
                         jobs=args.max_processes,
                         sent_model_path=args.sent_model_path,
                         batch_size=args.batch_size)
    AAS.run()
