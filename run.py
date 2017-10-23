# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
import argparse
import logging
import pickle
import shutil
import sys
from datetime import datetime
from os import makedirs, listdir, getcwd
from os.path import basename, exists, join, split, splitext, dirname
from time import time

import networkx as nx
import simplejson
from joblib import Parallel
from joblib import delayed

from aspects.rst.edu_tree_preprocesser import EDUTreePreprocesser
from aspects.aspects.edu_aspect_extractor import EDUAspectExtractor
from aspects.sentiment.sentiment_analyzer import \
    LogisticRegressionSentimentAnalyzer as SentimentAnalyzer
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.aspects.aspects_graph_builder import AspectsGraphBuilder
from aspects.results_analysis.results_analyzer import ResultsAnalyzer
from aspects.results_analysis.gerani_graph_analysis import get_dir_moi_for_node
from aspects.io.serializer import Serializer
from aspects.utilities.utils_multiprocess import batch_with_indexes
from aspects.utilities.custom_exceptions import WrongTypeException
from aspects.utilities.data_paths import IOPaths
from aspects.configs.conceptnets_config import CONCEPTNET_ASPECTS

sys.path.append('edu_dependency_parser/src/')
from parse import DiscourseParser

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s;%(filename)s:%(lineno)s;'
                           '%(funcName)s();%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='logs/run.log',
                    filemode='w',
                    )


def edu_parsing_multiprocess(parser, docs_id_range, edu_trees_dir,
                             extracted_documents_dir):
    processed = 0
    skipped = 0
    errors = 0

    n_docs = docs_id_range[1] - docs_id_range[0]

    for n_doc, document_id in enumerate(
            xrange(docs_id_range[0], docs_id_range[1]), start=1):
        start_time = datetime.now()

        logging.info(
            'EDU Parsing document id: {} -> {}/{}'.format(document_id, n_doc,
                                                          n_docs))
        try:
            edu_tree_path = edu_trees_dir + str(document_id) + '.tree'

            if exists(edu_tree_path):
                logging.info(
                    'EDU Tree Already exists: {}'.format(edu_tree_path))
                skipped += 1
            else:
                if parser is None:
                    parser = DiscourseParser(output_dir=edu_trees_dir,
                                             # verbose=True,
                                             # skip_parsing=True,
                                             # global_features=True,
                                             # save_preprocessed_doc=True,
                                             # preprocesser=None
                                             )

                document_path = extracted_documents_dir + str(document_id)

                if exists(document_path):
                    parser.parse(document_path)
                    # raise
                else:
                    logging.warning(
                        'Document #{} does not exist! Skipping to next one.'.format(
                            document_id))
                    errors += 1

                processed += 1
        # skip documents that parsing returns errors
        except (ValueError, IndexError, ZeroDivisionError, OSError) as err:
            logging.error(
                'Error for doc #{}: {}. It has been skipped'.format(
                    document_id, str(err)))
            if exists(edu_tree_path):
                shutil.rmtree(edu_tree_path)
            errors += 1

        logging.info(
            'EDU document id: {} -> parsed in {} seconds'.format(document_id, (
                datetime.now() - start_time).seconds))

    if parser is not None:
        parser.unload()

    logging.info(
        'Docs processed: {}, docs skipped: {}'.format(processed, skipped))


class AspectAnalysisSystem:
    def __init__(self, input_path, output_path, gold_standard_path,
                 analysis_results_path, jobs=1, sent_model_path=None,
                 n_logger=1000, batch_size=None):

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

    def _parse_input_documents(self):
        """
        Load and parse documents. All document should be stored in
        JSON/dictionary format, only values will be processed.

        @return:
            documents_count : int
                Number of documents processed
        """
        existing_documents_list = listdir(self.paths.extracted_docs)
        documents_count = len(existing_documents_list)

        # FIXME: disambiguate file loading and metadata information storing
        # todo why metadata is not stored?
        if documents_count == 0:
            f_extension = basename(self.input_file_path).split('.')[-1]
            logging.debug('Input file extension: {}'.format(f_extension))
            if f_extension in ['json']:
                with open(self.input_file_path, 'r') as f:
                    raw_documents = simplejson.load(f)
                    for ref_id, (doc_id, document) in enumerate(
                            raw_documents.iteritems()):
                        self.serializer.save(document, join(
                            self.paths.extracted_docs, str(
                                ref_id)))
                        self.serializer.save(str(doc_id), join(
                            self.paths.extracted_docs_ids, str(
                                ref_id)))
                        documents_count += 1
            # this is {'doc_id': {'text', text, 'metadata1': xxx}}
            # text with additional metadata
            elif f_extension in ['pkl', 'p', 'pickle']:
                with open(self.input_file_path, 'r') as f:
                    raw_documents = pickle.load(f)
                for ref_id, (doc_id, document) in enumerate(
                        raw_documents.iteritems()):
                    self.serializer.save(document['text'],
                                         join(self.paths.extracted_docs,
                                              str(ref_id)))
                    self.serializer.save({doc_id: document},
                                         join(
                                             self.paths.extracted_docs_metadata,
                                             str(ref_id)))
                    documents_count += 1
            elif f_extension in ['csv', 'txt']:
                raw_documents = {}
                with open(self.input_file_path, 'r') as f:
                    for idx, line in enumerate(f):
                        raw_documents[str(idx)] = line
                        self.serializer.save(line,
                                             self.paths.extracted_docs + str(
                                                 idx))
                        self.serializer.save({idx: line},
                                             self.paths.extracted_docs_metadata + str(
                                                 idx))
                        documents_count += 1
            else:
                raise WrongTypeException()
            logging.info('Number of all documents to analyse: {}'.format(
                len(raw_documents)))
        return documents_count

    def _perform_edu_parsing(self, documents_count, batch_size=None):
        logging.info('Documents: #{} will be processed'.format(documents_count))

        if batch_size is None:
            batch_size = documents_count / self.jobs
            if batch_size < 1:
                batch_size = 1
            logging.debug('Batch size for multiprocessing execution: {}'.format(
                batch_size))

        Parallel(n_jobs=self.jobs, verbose=5)(
            delayed(edu_parsing_multiprocess)(None, docs_id_range,
                                              self.paths.edu_trees,
                                              self.paths.extracted_docs)
            for docs_id_range, l in
            batch_with_indexes(range(documents_count), batch_size))

    def _perform_edu_preprocessing(self, documents_count):

        if not exists(self.paths.raw_edu_list):
            preprocesser = EDUTreePreprocesser()

            for document_id in range(0, documents_count):
                try:
                    if not document_id % self.n_loger:
                        logging.debug(
                            'EDU Preprocessor documentId: {}/{}'.format(
                                document_id, documents_count))
                    tree = self.serializer.load(
                        join(self.paths.edu_trees,
                             str(document_id) + '.tree.ser'))
                    preprocesser.processTree(tree, document_id)
                    self.serializer.save(tree, join(self.paths.link_trees,
                                                    str(document_id)))
                except TypeError as err:
                    logging.error(
                        'Document id: {} and error: {}'.format(document_id,
                                                               str(err)))
                    self.parsing_errors += 1
            edu_list = preprocesser.getPreprocessedEdusList()
            self.serializer.save(edu_list, self.paths.raw_edu_list)

    def _filter_edu_by_sentiment(self):
        """Filter out EDUs without sentiment, with neutral sentiment too"""

        if not (exists(self.paths.sentiment_filtered_edus)
                and exists(self.paths.docs_info)):

            if self.sent_model_path is None:
                analyzer = SentimentAnalyzer()
            else:
                analyzer = SentimentAnalyzer(model_path=self.sent_model_path)

            edu_list = list(
                self.serializer.load(self.paths.raw_edu_list).values())
            filtered_edus = {}
            dos_info = {}

            for edu_id, edu in enumerate(edu_list):
                edu['sentiment'] = []
                logging.debug('edu: {}'.format(edu))
                sentiment = analyzer.analyze(edu['raw_text'])[0]

                # todo add to readme strucutre of docuemnt info and other dicts
                if not edu['source_document_id'] in dos_info:
                    dos_info[edu['source_document_id']] = {
                        'EDUs': [],
                        'accepted_edus': [],
                        'aspects': {},
                        'aspect_concepts': {},
                        'aspect_keywords': {},
                        'sentiment': {},
                    }

                dos_info[edu['source_document_id']]['sentiment'].update(
                    {edu_id: sentiment})
                dos_info[edu['source_document_id']]['EDUs'].append(edu_id)

                if sentiment:
                    edu['sentiment'].append(sentiment)
                    dos_info[edu['source_document_id']][
                        'accepted_edus'].append(edu_id)

                    filtered_edus[edu_id] = edu

            self.serializer.save(filtered_edus,
                                 self.paths.sentiment_filtered_edus)
            self.serializer.save(dos_info, self.paths.docs_info)

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
        max_edu_id = max(edus.keys())

        logging.info('# of document with sentiment edus: {}'.format(n_edus))

        for eduid, edu in edus.iteritems():
            if eduid not in aspects_per_edu:
                doc_info = documents_info[edu['source_document_id']]
                aspects, aspect_concepts, aspect_keywords = extractor.extract(
                    edu)
                aspects_per_edu[eduid] = aspects
                logging.info(
                    'EDU ID/MAX EDU ID: {}/{}'.format(eduid, max_edu_id))
                logging.debug(
                    'aspects: {}'.format(aspects))

                if 'aspects' not in doc_info:
                    doc_info['aspects'] = []

                doc_info['aspects'].update({eduid: aspects})
                doc_info['aspect_concepts'].update({eduid: aspect_concepts})
                doc_info['aspect_keywords'].update({eduid: aspect_keywords})

                if not eduid % 100:
                    logging.info(
                        'Save partial aspects, edu_id {}'.format(eduid))
                    self.serializer.save(aspects_per_edu,
                                         self.paths.aspects_per_edu)
                    self.serializer.save(documents_info,
                                         self.paths.docs_info)

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

            for doc_id, doc_info in docs_info.iteritems():

                if len(doc_info['accepted_edus']) > 0:
                    link_tree = self.serializer.load(
                        join(self.paths.link_trees, str(doc_id)))

                extracted_rules = rules_extractor.extract(link_tree,
                                                          doc_info[
                                                              'accepted_edus'],
                                                          doc_id)

                # if len(extracted_rules) > 0:
                rules.update(extracted_rules)

            logging.info('Rules extracted.')
            self.serializer.save(rules, self.paths.edu_dependency_rules)

    def _build_aspect_dependency_graph(self):
        """Build dependency graph"""

        if not (exists(self.paths.aspects_graph) and exists(
                self.paths.aspects_importance)):
            dependency_rules = self.serializer.load(
                self.paths.edu_dependency_rules)
            aspects_per_edu = self.serializer.load(
                self.paths.aspects_per_edu)
            documents_info = self.serializer.load(
                self.paths.docs_info)

            builder = AspectsGraphBuilder(aspects_per_edu)
            graph, page_ranks = builder.build(dependency_rules,
                                              documents_info,
                                              CONCEPTNET_ASPECTS,
                                              filter_confidence=1,
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
                        aspect_position = float(
                            aspects_list.index(aspect) + 1) / aspects_count
                        if aspect_position < threshold:
                            aspects.append(aspect)
            documents_info[documentId]['aspects'] = aspects
        self.serializer.save(documents_info, self.paths.final_docs_info)

    def _analyze_results(self, threshold):
        """ remove noninformative aspects  """

        documents_info = self.serializer.load(
            self.paths.final_docs_info)
        gold_standard = self.serializer.load(self.gold_standard_path)

        if gold_standard is None:
            raise ValueError('GoldStandard data is None')

        analyzer = ResultsAnalyzer()

        for document_id, document_info in documents_info.iteritems():
            # todo: why document is never used?
            document = self.serializer.load(
                join(self.paths.extracted_docs + str(document_id)))
            analyzer.analyze(document_info['aspects'],
                             gold_standard[document_id])
        measures = analyzer.get_analysis_results()
        self.serializer.append_serialized(
            ';'.join(str(x) for x in [threshold] + measures) + '\n',
            self.analysis_results_path)

    def _add_sentiment_to_graph_adn_dir_moi(self):
        aspects_per_edu = self.serializer.load(self.paths.aspects_per_edu)
        documents_info = self.serializer.load(self.paths.docs_info)
        aspect_graph = self.serializer.load(self.paths.aspects_graph)
        aspect_graph = get_dir_moi_for_node(aspect_graph, aspects_per_edu,
                                            documents_info)
        self.serializer.save(aspect_graph, self.paths.aspects_graph)

    def run(self):

        total_timer_start = time()

        # load documents
        logging.info('--------------------------------------')
        logging.info("Extracting documents from input file...")

        timer_start = time()
        documents_count = self._parse_input_documents()
        timer_end = time()

        logging.info("Extracted", documents_count,
                     "documents from input file in {:.2f} seconds.".format(
                         timer_end - timer_start))

        # preprocessing and rhetorical parsing
        logging.info('--------------------------------------')
        logging.info("Performing EDU segmentation and dependency parsing...")

        timer_start = time()
        self._perform_edu_parsing(documents_count, batch_size=self.batch_size)
        timer_end = time()

        logging.info("EDU segmentation and dependency parsing documents "
                     "from input file in {:.2f} seconds.".format(
            timer_end - timer_start))

        # process EDU based on rhetorical trees
        logging.info('--------------------------------------')
        logging.info("Performing EDU trees preprocessing in {:.2f} seconds.".
                     format(timer_end - timer_start))

        timer_start = time()
        self._perform_edu_preprocessing(documents_count)
        logging.warning('{} trees were not parse!'.format(self.parsing_errors))
        timer_end = time()

        logging.info(
            "EDU trees preprocessing succeeded in {:.2f} seconds".format(
                timer_end - timer_start))

        # filter EDU with sentiment orientation only
        logging.info('--------------------------------------')
        logging.info("Performing EDU sentiment filtering...")

        timer_start = time()
        self._filter_edu_by_sentiment()
        timer_end = time()

        logging.info("EDU filtering succeeded in {:.2f} seconds".format(
            timer_end - timer_start))

        # extract aspects
        logging.info('--------------------------------------')
        logging.info("Performing EDU aspects extraction...")

        timer_start = time()
        self._extract_aspects_from_edu()
        timer_end = time()

        logging.info("EDU aspects extraction in {:.2f} seconds".format(
            timer_end - timer_start))

        # rule extraction
        logging.info('--------------------------------------')
        logging.info("Performing EDU dependency rules extraction...")

        timer_start = time()
        self._extract_edu_dependency_rules()
        timer_end = time()

        logging.info("EDU dependency rules extraction succeeded "
                     "in {:.2f} seconds".format(timer_end - timer_start))

        # build aspect-aspect graph
        logging.info('--------------------------------------')
        logging.info("Performing aspects graph building...")

        timer_start = time()
        self._build_aspect_dependency_graph()
        timer_end = time()

        logging.info(
            "Aspects graph building succeeded in {:.2f} seconds".format(
                timer_end - timer_start))

        # add sentiments to nodes/aspects and count Gerani dir-moi weight
        logging.info('--------------------------------------')
        logging.info("Sentiments to nodes/aspects and Gerani dir-moi weight...")

        timer_start = time()
        self._add_sentiment_to_graph_adn_dir_moi()
        timer_end = time()

        logging.info(
            "Graph extended with sentiments for nodes and dir-moi in "
            "{:.2f} seconds".format(timer_end - timer_start))

        logging.info('--------------------------------------')
        total_timer_end = time()

        logging.info("Whole system run in {:.2f} seconds".format(
            total_timer_end - total_timer_start))

        logging.info('--------------------------------------')
        logging.info('Save graph with Gephi suitable extension')
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
    AAS = AspectAnalysisSystem(input_path=args.input_file_path,
                               output_path=output_path,
                               gold_standard_path=gold_standard_path,
                               analysis_results_path=args.analysis_results_path,
                               jobs=args.max_processes,
                               sent_model_path=args.sent_model_path,
                               batch_size=args.batch_size)
    AAS.run()
