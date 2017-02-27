# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda
import pickle
import sys
import shutil
import subprocess

import argparse
from joblib import Parallel
from joblib import delayed
import simplejson
import os.path
from os.path import basename, exists
from pprint import pprint
from time import time
from datetime import datetime
from optparse import OptionParser

from tqdm import tqdm

sys.path.append('edu_dependency_parser/src/')
from parse import DiscourseParser
from EDUTreePreprocesser import EDUTreePreprocesser
from EDUAspectExtractor import EDUAspectExtractor
from LogisticRegressionSentimentAnalyzer import \
    LogisticRegressionSentimentAnalyzer as SentimentAnalyzer
from EDUTreeRulesExtractor import EDUTreeRulesExtractor
from AspectsGraphBuilder import AspectsGraphBuilder
from ResultsAnalyzer import ResultsAnalyzer
from Serializer import Serializer
from utils_multiprocess import batch_with_indexes

import logging
import sys

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


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

            if os.path.exists(edu_tree_path):
                logging.info(
                    'EDU Tree Already exists: {}'.format(edu_tree_path))
                skipped += 1
            else:
                if parser is None:
                    # parser = DiscourseParser(output_dir=edu_trees_dir)
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
    def __init__(self, input_path, output_path, gold_standard_path, jobs=1,
                 sent_model_path=None,
                 n_logger=1000, batch_size=None):

        self.input_file_path = input_path
        self.batch_size = batch_size
        self.__ensurePathExist(output_path)

        self.sent_model_path = sent_model_path

        self.paths = {}
        self.paths['input'] = input_path

        self.paths[
            'extracted_documents_dir'] = output_path + '/extracted_documents/'
        self.__ensurePathExist(self.paths['extracted_documents_dir'])

        self.paths[
            'extracted_documents_ids'] = output_path + '/extracted_documents_ids/'
        self.__ensurePathExist(self.paths['extracted_documents_ids'])

        self.paths[
            'extracted_documents_metadata'] = output_path + '/extracted_documents_metadata/'
        self.__ensurePathExist(self.paths['extracted_documents_metadata'])

        self.paths['documents_info'] = output_path + '/documents_info'

        self.paths['edu_trees_dir'] = output_path + '/edu_trees_dir/'
        self.__ensurePathExist(self.paths['edu_trees_dir'])

        self.paths['link_trees_dir'] = output_path + '/link_trees_dir/'
        self.__ensurePathExist(self.paths['link_trees_dir'])

        self.paths['raw_edu_list'] = output_path + '/raw_edu_list'
        self.paths[
            'sentiment_filtered_edus'] = output_path + '/sentiment_filtered_edus'
        self.paths['aspects_per_edu'] = output_path + '/aspects_per_edu'
        self.paths[
            'edu_dependency_rules'] = output_path + '/edu_dependency_rules'

        self.paths['aspects_graph'] = output_path + '/aspects_graph'
        self.paths[
            'aspects_importance'] = output_path + '/aspects_importance'

        self.paths[
            'final_documents_info'] = output_path + '/final_documents_info'

        self.paths['gold_standard'] = gold_standard_path

        self.paths['results'] = output_path + '/results_allGraph_filter1.csv'

        self.serializer = Serializer()

        # number of all processes
        self.jobs = jobs

        # by how many examples logging will be done
        self.n_loger = n_logger

        # count number of error within parsing RDT
        self.parsing_errors = 0

    #
    #   Sprawdza czy sciezka istnieje i tworzy ja w razie potrzeby
    #
    def __ensurePathExist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    #
    #   Parsowanie dokumentow wejsciowych
    #
    def __parse_input_documents(self):
        """
        Load and parse documents. All document should be stored in JSON/dictionary format,
        only values will be processed.
        @return:
            documents_count : int
                Number of documents processed
        """
        existing_documents_list = os.listdir(
            self.paths['extracted_documents_dir'])
        documents_count = len(existing_documents_list)

        # FIXME: disambiguate file loading and metadata information storing
        if documents_count == 0:
            f_extension = basename(self.input_file_path).split('.')[-1]
            logging.debug('Input file extension: {}'.format(f_extension))
            if f_extension in ['json']:
                with open(self.input_file_path, 'r') as f:
                    raw_documents = simplejson.load(f)
                    for ref_id, (doc_id, document) in enumerate(
                            raw_documents.iteritems()):
                        self.serializer.save(document, self.paths[
                            'extracted_documents_dir'] + str(ref_id))
                        self.serializer.save(str(doc_id), self.paths[
                            'extracted_documents_ids'] + str(ref_id))
                        documents_count += 1
            # this is {'doc_id': {'text', text, 'metadata1': xxx}}
            # text with additional metadata
            elif f_extension in ['pkl', 'p', 'pickle']:
                with open(self.input_file_path, 'r') as f:
                    raw_documents = pickle.load(f)
                    print raw_documents.items()[:2]
                for ref_id, (doc_id, document) in enumerate(
                        raw_documents.iteritems()):
                    self.serializer.save(document['text'], self.paths[
                        'extracted_documents_dir'] + str(ref_id))
                    self.serializer.save({doc_id: document}, self.paths[
                        'extracted_documents_metadata'] + str(ref_id))
                    documents_count += 1
            # elif f_extension in ['txt', 'csv']:
            #     f = open(inputFilePath, "r")
            #     input_ = f.read()
            #     print input_
            #     raw_documents = input_.split('\n\n')
            #     f.close()
            else:
                raise 'Wrong file type -> extension'
            # for doc_id, document in enumerate(raw_documents):
            #     self.serializer.save(document, self.paths[
            #         'extracted_documents_dir'] + str(doc_id))
            #     self.serializer.save(document, self.paths[
            #         'extracted_documents_ids'] + str(doc_id))
            #     documents_count += 1
            logging.info('Number of all documents to analyse: {}'.format(
                len(raw_documents)))
        return documents_count

    #
    #   Parsowanie dokumentow na drzewa EDU
    #
    def __perform_edu_parsing(self, documents_count, batch_size=None):
        # parser = None
        # processed = 0
        # skipped = 0

        logging.info('Documents: #{} will be processed'.format(documents_count))

        if batch_size is None:
            batch_size = documents_count / self.jobs
            if batch_size < 1:
                batch_size = 1
            logging.debug('Batch size for multiprocessing execution: {}'.format(
                batch_size))

        Parallel(n_jobs=self.jobs, verbose=5)(
            delayed(edu_parsing_multiprocess)(None, docs_id_range,
                                              self.paths['edu_trees_dir'],
                                              self.paths[
                                                  'extracted_documents_dir'])
            for docs_id_range, l in
            batch_with_indexes(range(documents_count), batch_size))

    # for documentId in range(0, documentsCount):
    # 	logging.debug('__performEDUParsing documentId: {}'.format(documentId))
    # 	EDUTreePath = self.paths['edu_trees_dir'] + str(
    # 		documentId) + '.tree'
    #
    # 	if os.path.exists(EDUTreePath):
    # 		print 'EDU Tree Already exists: {}'.format(EDUTreePath)
    # 		skipped += 1
    # 	else:
    # 		if parser is None:
    # 			parser = DiscourseParser(
    # 				output_dir=self.paths['edu_trees_dir'])
    #
    # 		documentPath = self.paths['extracted_documents_dir'] + str(
    # 			documentId)
    # 		parser.parse(documentPath)
    #
    # 		processed += 1

    # if parser is not None:
    # 	parser.unload()

    # return processed, skipped

    #
    #   Preprocessing danych - oddzielanie zalezno�ci EDU od tekstu
    #
    def __performEDUPreprocessing(self, documents_count):

        if not os.path.exists(self.paths['raw_edu_list']):
            preprocesser = EDUTreePreprocesser()
            # Parallel(n_jobs=self.jobs)(
            # 	delayed(self.__performEDUPreprocessing_multiprocess())(preprocesser, docs_id_range)
            # 	for docs_id_range, in batch_with_indexes(range(documentsCount), self.jobs))
            for document_id in range(0, documents_count):
                try:
                    if not document_id % self.n_loger:
                        logging.debug(
                            'EDU Preprocessor documentId: {}/{}'.format(
                                document_id, documents_count))
                    tree = self.serializer.load(
                        self.paths['edu_trees_dir'] + str(
                            document_id) + '.tree.ser')
                    preprocesser.processTree(tree, document_id)
                    self.serializer.save(tree,
                                         self.paths['link_trees_dir'] + str(
                                             document_id))
                except TypeError as err:
                    logging.error(
                        'Document id: {} and error: {}'.format(document_id,
                                                               str(err)))
                    self.parsing_errors += 1
            edu_list = preprocesser.getPreprocessedEdusList()
            self.serializer.save(edu_list, self.paths['raw_edu_list'])

    # def __performEDUPreprocessing_multiprocess(self, preprocesser, docs_id_range):
    #
    # 	for document_id in range(docs_id_range[0], docs_id_range[1]):
    # 		logging.debug(
    # 			'__performEDUPreprocessing documentId: {}'.format(document_id))
    # 		tree = self.serializer.load(self.paths['edu_trees_dir'] + str(
    # 			document_id) + '.tree.ser')
    #
    # 		preprocesser.processTree(tree, document_id)
    #
    # 		self.serializer.save(tree, self.paths['link_trees_dir'] + str(
    # 			document_id))

    #
    #   Analiza sentymentu EDU i odsianie niesentymentalnych
    #
    def __filterEDUBySentiment(self):

        if not (os.path.exists(
                self.paths['sentiment_filtered_edus']) and os.path.exists(
            self.paths['documents_info'])):

            if self.sent_model_path is None:
                analyzer = SentimentAnalyzer()
            else:
                analyzer = SentimentAnalyzer(model_path=self.sent_model_path)

            EDUList = list(
                self.serializer.load(self.paths['raw_edu_list']).values())
            # logging.debug('EDU List: {}'.format(EDUList))
            # pprint(EDUList)
            filteredEDUs = {}
            documentsInfo = {}

            for EDUId, EDU in enumerate(EDUList):
                # logging.debug('EDU: {}'.format(EDU))
                sentiment = analyzer.analyze(EDU['raw_text'])

                if not EDU['source_document_id'] in documentsInfo:
                    documentsInfo[EDU['source_document_id']] = {'sentiment': 0,
                                                                'EDUs': [],
                                                                'accepted_edus': []}

                documentsInfo[EDU['source_document_id']][
                    'sentiment'] += sentiment
                documentsInfo[EDU['source_document_id']]['EDUs'].append(EDUId)

                if (sentiment <> 0):
                    EDU['sentiment'] = sentiment
                    documentsInfo[EDU['source_document_id']][
                        'accepted_edus'].append(EDUId)

                    filteredEDUs[EDUId] = EDU

            self.serializer.save(filteredEDUs,
                                 self.paths['sentiment_filtered_edus'])
            self.serializer.save(documentsInfo, self.paths['documents_info'])

    #
    #   Ekstrakcja aspekt�w
    #
    def __extractAspectsFromEDU(self):
        if not os.path.exists(self.paths['aspects_per_edu']):

            # EDUs = self.serializer.load(self.paths['raw_edu_list'])
            EDUs = self.serializer.load(self.paths['sentiment_filtered_edus'])
            documentsInfo = self.serializer.load(self.paths['documents_info'])

            aspectsPerEDU = {}

            extractor = EDUAspectExtractor()

            for EDUId, EDU in EDUs.iteritems():
                asp = extractor.extract(EDU)
                aspectsPerEDU[EDUId] = asp
                # logging.debug('Aspect: {}'.format(asp))

                # if not 'aspects' in documentsInfo[EDU['source_document_id']]:
                if 'aspects' not in documentsInfo[EDU['source_document_id']]:
                    documentsInfo[EDU['source_document_id']]['aspects'] = []

                documentsInfo[EDU['source_document_id']][
                    'aspects'] = extractor.getAspectsInDocument(
                    documentsInfo[EDU['source_document_id']]['aspects'],
                    aspectsPerEDU[EDUId])

            self.serializer.save(aspectsPerEDU, self.paths['aspects_per_edu'])
            self.serializer.save(documentsInfo, self.paths['documents_info'])

    #
    #   Ekstrakcja regu� asocjacyjnych z drzewa zaleznosci EDU
    #
    def __extractEDUDepencencyRules(self, documentsCount):

        if not os.path.exists(self.paths['edu_dependency_rules']):

            rulesExtractor = EDUTreeRulesExtractor()
            rules = []

            documentsInfo = self.serializer.load(self.paths['documents_info'])

            for documentId, documentInfo in documentsInfo.iteritems():

                if len(documentInfo['accepted_edus']) > 0:
                    linkTree = self.serializer.load(
                        self.paths['link_trees_dir'] + str(documentId))

                extractedRules = rulesExtractor.extract(linkTree, documentInfo[
                    'accepted_edus'])

                if len(extractedRules) > 0:
                    rules += extractedRules

            self.serializer.save(rules, self.paths['edu_dependency_rules'])

    #
    #   Budowa grafu zale�no�ci aspekt�w
    #
    def __buildAspectDepencencyGraph(self):

        if not (os.path.exists(self.paths['aspects_graph']) and os.path.exists(
                self.paths['aspects_importance'])):
            dependencyRules = self.serializer.load(
                self.paths['edu_dependency_rules'])
            aspectsPerEDU = self.serializer.load(self.paths['aspects_per_edu'])

            builder = AspectsGraphBuilder()
            graph, pageRanks = builder.build(dependencyRules, aspectsPerEDU)

            self.serializer.save(graph, self.paths['aspects_graph'])
            self.serializer.save(pageRanks, self.paths['aspects_importance'])

    #
    #   Odsiewamy �mieciowe aspekty na podsawie informacji o ich wa�nosci
    #
    def __filterAspects(self, threshold):

        aspectsImportance = self.serializer.load(
            self.paths['aspects_importance'])
        documentsInfo = self.serializer.load(self.paths['documents_info'])

        aspectsCount = len(aspectsImportance)
        aspectsList = list(aspectsImportance)

        # """
        for documentId, documentInfo in documentsInfo.iteritems():

            aspects = []

            if 'aspects' in documentInfo:

                for aspect in documentInfo['aspects']:
                    if aspect in aspectsImportance:
                        aspectPosition = float(
                            aspectsList.index(aspect) + 1) / aspectsCount

                        if aspectPosition < threshold:
                            aspects.append(aspect)

            documentsInfo[documentId]['aspects'] = aspects

        # print '--------------------'
        # pprint(documentsInfo)
        self.serializer.save(documentsInfo, self.paths['final_documents_info'])
        """

        aspectsPerEDU = self.serializer.load(self.paths['aspects_per_edu'])

        for documentId, documentInfo in documentsInfo.iteritems():

            aspects = []

            for EDUId in documentInfo['accepted_edus']:
                mainAspect = None
                mainAspectImportance = -1

                for aspect in aspectsPerEDU[EDUId]:
                    if aspect in aspectsImportance:

                        aspectPosition = float(aspectsList.index(aspect)+1)/aspectsCount

                        if aspectPosition < threshold and aspectsImportance[aspect] > mainAspectImportance:
                            mainAspect = aspect
                            mainAspectImportance = aspectsImportance[aspect]

                if mainAspect is not None:
                    aspects.append(mainAspect)


            documentsInfo[documentId]['aspects'] = aspects

        self.serializer.save(documentsInfo, self.paths['final_documents_info'])
        #
        """

    #
    #   Odsiewamy �mieciowe aspekty na podsawie informacji o ich wa�nosci
    #
    def __analyzeResults(self, threshold):

        documentsInfo = self.serializer.load(self.paths['final_documents_info'])
        goldStandard = self.serializer.load(self.paths['gold_standard'])

        if goldStandard is None:
            raise ValueError('GoldStandard data is None')

        analyzer = ResultsAnalyzer()

        for documentId, documentInfo in documentsInfo.iteritems():
            document = self.serializer.load(
                self.paths['extracted_documents_dir'] + str(documentId))

            analyzer.analyze(documentInfo['aspects'], goldStandard[documentId])

        measures = analyzer.getAnalyzisResults()

        self.serializer.append(
            ';'.join(str(x) for x in [threshold] + measures) + '\n',
            self.paths['results'])

    def run(self):

        total_timer_start = time()

        # load documents
        logging.info('--------------------------------------')
        logging.info("Extracting documents from input file...")

        timer_start = time()
        documents_count = self.__parse_input_documents()
        timer_end = time()

        logging.info("Extracted", documents_count,
                     "documents from input file in {:.2f} seconds.".format(
                         timer_end - timer_start))

        # preprocessing and rhetorical parsing
        logging.info('--------------------------------------')
        logging.info("Performing EDU segmentation and dependency parsing...")

        timer_start = time()
        self.__perform_edu_parsing(documents_count, batch_size=self.batch_size)
        timer_end = time()

        # process EDU based on rhetorical trees
        logging.info('--------------------------------------')
        logging.info("Performing EDU trees preprocessing...")

        timer_start = time()
        self.__performEDUPreprocessing(documents_count)
        logging.warning('{} trees were not parse!'.format(self.parsing_errors))
        timer_end = time()

        logging.info(
            "EDU trees preprocessing succeeded in {:.2f} seconds".format(
                timer_end - timer_start))

        # filter EDU with sentiment orientation only
        logging.info('--------------------------------------')
        logging.info("Performing EDU sentiment filtering...")

        timer_start = time()
        self.__filterEDUBySentiment()
        timer_end = time()

        logging.info("EDU filtering succeeded in {:.2f} seconds".format(
            timer_end - timer_start))

        # extract aspects
        logging.info('--------------------------------------')
        logging.info("Performing EDU aspects extraction...")

        timer_start = time()
        self.__extractAspectsFromEDU()
        timer_end = time()

        logging.info("EDU aspects extraction in {:.2f} seconds".format(
            timer_end - timer_start))

        # rule extraction
        logging.info('--------------------------------------')
        logging.info("Performing EDU dependency rules extraction...")

        timer_start = time()
        self.__extractEDUDepencencyRules(documents_count)
        timer_end = time()

        logging.info(
            "EDU dependency rules extraction succeeded in {:.2f} seconds".format(
                timer_end - timer_start))

        # build aspect-aspect graph
        logging.info('--------------------------------------')
        logging.info("Performing aspects graph building...")

        timer_start = time()
        self.__buildAspectDepencencyGraph()
        timer_end = time()

        logging.info(
            "Aspects graph building succeeded in {:.2f} seconds".format(
                timer_end - timer_start))

        # for i in range(1, 1000):
        #     threshold = i / 1000.0

        # filter aspects
        logging.info('--------------------------------------')
        # logging.info("Performing aspects filtering with threshold: {}".format(
        #     threshold))
        #
        # timer_start = time()
        # self.__filterAspects(threshold)
        # timer_end = time()
        #
        # logging.info("Aspects filtering succeeded in {:.2f} seconds".format(
        #     timer_end - timer_start))
        #
        # # results analysis
        # logging.info('--------------------------------------')
        # logging.info("Performing results analysis...")
        #
        # timer_start = time()
        # self.__analyzeResults(threshold)
        # timer_end = time()
        #
        # logging.info("Results analysis succeeded in {:.2f} seconds".format(
        # timer_end - timer_start))

        total_timer_end = time()

        logging.info("Whole system run in {.2f} seconds".format(
            total_timer_end - total_timer_start))


if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    DEFAULT_OUTPUT_PATH = os.path.join(ROOT_PATH, 'results')
    DEFAULT_INPUT_FILE_PATH = os.path.join(ROOT_PATH, 'texts', 'test.txt')

    arg_parser = argparse.ArgumentParser(description='Process documents.')
    arg_parser.add_argument('-input', type=str, dest='input_file_path',
                            default=DEFAULT_INPUT_FILE_PATH,
                            help='Path to the file with documents (json, csv, pickle)')
    arg_parser.add_argument('-output', type=str, dest='output_file_path',
                            default=DEFAULT_OUTPUT_PATH,
                            help='Number of processes')
    arg_parser.add_argument('-sent_model', type=str, dest='sent_model_path',
                            default=None,
                            help='path to sentiment model')
    arg_parser.add_argument('-batch', type=int, dest='batch_size', default=None,
                            help='batch size for each process')
    arg_parser.add_argument('-p', type=int, dest='max_processes', default=1,
                            help='Number of processes')
    args = arg_parser.parse_args()

    input_file_full_name = os.path.split(args.input_file_path)[1]
    input_file_name = os.path.splitext(input_file_full_name)[0]
    output_path = os.path.join(args.output_file_path, input_file_name)
    gold_standard_path = os.path.dirname(
        args.input_file_path) + input_file_name + '_aspects_list.ser'
    AAS = AspectAnalysisSystem(input_path=args.input_file_path,
                               output_path=output_path,
                               gold_standard_path=gold_standard_path,
                               jobs=args.max_processes,
                               sent_model_path=args.sent_model_path,
                               batch_size=args.batch_size)
    AAS.run()
