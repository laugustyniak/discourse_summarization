# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

import sys
import shutil
import subprocess
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
from LogisticRegressionSentimentAnalyzer import LogisticRegressionSentimentAnalyzer as SentimentAnalyzer
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

    for n_doc, document_id in enumerate(xrange(docs_id_range[0], docs_id_range[1]), start=1):
        start_time = datetime.now()

        logging.info('EDU Parsing document id: {} -> {}/{}'.format(document_id, n_doc, n_docs))
        try:
            edu_tree_path = edu_trees_dir + str(document_id) + '.tree'

            if os.path.exists(edu_tree_path):
                logging.info('EDU Tree Already exists: {}'.format(edu_tree_path))
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
                    logging.warning('Document #{} does not exist! Skipping to next one.'.format(document_id))
                    errors += 1

                processed += 1
        except (ValueError, IndexError) as err:
            logging.error('ValueError for doc #{}: {}'.format(document_id, str(err)))
            shutil.rmtree(edu_tree_path)
            errors += 1
            # raise ValueError
        # except IndexError as err:
        #     logging.error('IndexError for doc #{}: {}'.format(document_id, str(err)))
        #     # shutil.rmtree(edu_tree_path)
        #     pass

        logging.info(
            'EDU document id: {} -> parsed in {} seconds'.format(document_id, (datetime.now() - start_time).seconds))

    if parser is not None:
        parser.unload()

    logging.info(
        'Docs processed: {}, docs skipped: {}'.format(processed, skipped))


class AspectAnalysisSystem:
    def __init__(self, inputPath, outputPathRoot, goldStandardPath, jobs=1, sent_model_path=None,
                 n_logger=1000):

        self.__ensurePathExist(outputPathRoot)

        self.sent_model_path = sent_model_path

        self.paths = {}
        self.paths['input'] = inputPath

        self.paths[
            'extracted_documents_dir'] = outputPathRoot + '/extracted_documents/'
        self.__ensurePathExist(self.paths['extracted_documents_dir'])

        self.paths[
            'extracted_documents_ids'] = outputPathRoot + '/extracted_documents_ids/'
        self.__ensurePathExist(self.paths['extracted_documents_ids'])

        self.paths['documents_info'] = outputPathRoot + '/documents_info'

        self.paths['edu_trees_dir'] = outputPathRoot + '/edu_trees_dir/'
        self.__ensurePathExist(self.paths['edu_trees_dir'])

        self.paths['link_trees_dir'] = outputPathRoot + '/link_trees_dir/'
        self.__ensurePathExist(self.paths['link_trees_dir'])

        self.paths['raw_edu_list'] = outputPathRoot + '/raw_edu_list'
        self.paths[
            'sentiment_filtered_edus'] = outputPathRoot + '/sentiment_filtered_edus'
        self.paths['aspects_per_edu'] = outputPathRoot + '/aspects_per_edu'
        self.paths[
            'edu_dependency_rules'] = outputPathRoot + '/edu_dependency_rules'

        self.paths['aspects_graph'] = outputPathRoot + '/aspects_graph'
        self.paths[
            'aspects_importance'] = outputPathRoot + '/aspects_importance'

        self.paths[
            'final_documents_info'] = outputPathRoot + '/final_documents_info'

        self.paths['gold_standard'] = goldStandardPath

        self.paths['results'] = outputPathRoot + '/results_allGraph_filter1.csv'

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

        if documents_count == 0:
            f_extension = basename(inputFilePath).split('.')[-1]
            logging.debug('Input file extension: {}'.format(f_extension))
            if f_extension in ['json']:
                with open(inputFilePath, 'r') as f:
                    raw_documents = simplejson.load(f)
                    for ref_id, (doc_id, document) in enumerate(raw_documents.iteritems()):
                        self.serializer.save(document, self.paths[
                            'extracted_documents_dir'] + str(ref_id))
                        self.serializer.save(str(doc_id), self.paths[
                            'extracted_documents_ids'] + str(ref_id))
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
            logging.debug('Batch size for multiprocessing execution: {}'.format(batch_size))

        Parallel(n_jobs=self.jobs, verbose=5)(
            delayed(edu_parsing_multiprocess)(None, docs_id_range,
                                              self.paths['edu_trees_dir'],
                                              self.paths['extracted_documents_dir'])
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
                        logging.debug('EDU Preprocessor documentId: {}/{}'.format(document_id, documents_count))
                    tree = self.serializer.load(self.paths['edu_trees_dir'] + str(document_id) + '.tree.ser')
                    preprocesser.processTree(tree, document_id)
                    self.serializer.save(tree, self.paths['link_trees_dir'] + str(document_id))
                except TypeError as err:
                    logging.error('Document id: {} and error: {}'.format(document_id, str(err)))
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
                self.paths['sentiment_filtered_edus']) and os.path.exists(self.paths['documents_info'])):

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

        totalTimerStart = time()

        # wczytujemy dokumenty wejsciowe

        print '--------------------------------------'
        print "Extracting documents from input file..."

        timerStart = time()
        documentsCount = self.__parse_input_documents()
        timerEnd = time()

        print "Extracted", documentsCount, "documents from input file in %.2f seconds." % (
            timerEnd - timerStart)

        #   przeprowadzamy analiz� dyskursu - segmentujemy na edu i parsujemy zaleznosci
        print '--------------------------------------'
        print "Performing EDU segmentation and dependency parsing..."

        timerStart = time()
        self.__perform_edu_parsing(documentsCount)
        timerEnd = time()

        # print "EDU dependency parsing succeeded in %.2f seconds. Processed" % (
        # 	timerEnd - timerStart), processed, "documents, skipped", skipped, "documents"

        #   przeprowadzamy preprocessing danych na drzewie EDU
        print '--------------------------------------'
        print "Performing EDU trees preprocessing..."

        timerStart = time()
        self.__performEDUPreprocessing(documentsCount)
        logging.warning('{} trees were not parse!'.format(self.parsing_errors))
        timerEnd = time()

        print "EDU trees preprocessing succeeded in %.2f seconds" % (
            timerEnd - timerStart)

        #   przeprowadzamy filtrowanie EDU wg sentymentu
        print '--------------------------------------'
        print "Performing EDU sentiment filtering..."

        timerStart = time()
        self.__filterEDUBySentiment()
        timerEnd = time()

        print "EDU filtering succeeded in %.2f seconds" % (
            timerEnd - timerStart)

        #   przeprowadzamy ekstrakcj� aspekt�w z EDU
        print '--------------------------------------'
        print "Performing EDU aspects extraction..."

        timerStart = time()
        self.__extractAspectsFromEDU()
        timerEnd = time()

        print "EDU aspects extraction in %.2f seconds" % (timerEnd - timerStart)

        #   przeprowadzamy ekstrakcj� regu� zale�no�ci pomi�dzy EDU
        print '--------------------------------------'
        print "Performing EDU dependency rules extraction..."

        timerStart = time()
        self.__extractEDUDepencencyRules(documentsCount)
        timerEnd = time()

        print "EDU dependency rules extraction succeeded in %.2f seconds" % (
            timerEnd - timerStart)

        #   przeprowadzamy budow� grafu aspekt�w
        print '--------------------------------------'
        print "Performing aspects graph building..."

        timerStart = time()
        self.__buildAspectDepencencyGraph()
        timerEnd = time()

        print "Aspects graph building succeeded in %.2f seconds" % (
            timerEnd - timerStart)

        for i in range(1, 1000):
            threshold = i / 1000.0

        # filtrujemy aspekty
        # print '--------------------------------------'
        # print "Performing aspects filtering...", threshold

        # timerStart = time()
        # self.__filterAspects(threshold)
        # timerEnd = time()

        # print "Aspects filtering succeeded in %.2f seconds" % (timerEnd - timerStart)

        # analizujemy wyniki
        # print '--------------------------------------'
        # print "Performing results analysis..."
        #
        # timerStart = time()
        # self.__analyzeResults(threshold)
        timerEnd = time()

        # print "Results analysis succeeded in %.2f seconds" % (timerEnd - timerStart)

        totalTimerEnd = time()

        print "Whole system run in %.2f seconds" % (
            totalTimerEnd - totalTimerStart)


if __name__ == "__main__":

    ROOT_PATH = os.getcwd()
    OUTPUT_PATH = os.path.join('/datasets/sentiment/aspects', 'results/')
    # OUTPUT_PATH = os.path.join(ROOT_PATH, 'results/')
    INPUT_PATH = os.path.join(ROOT_PATH, 'edu_dependency_parser/texts/')
    DEFAULT_INPUT_FILENAME = 'test.txt'

    sysArgs = sys.argv[1:]
    sent_model_path = None

    if len(sysArgs) == 0:
        inputFilePath = INPUT_PATH + DEFAULT_INPUT_FILENAME

        print "No input file specified. Using default: " + INPUT_PATH + DEFAULT_INPUT_FILENAME

    else:
        inputFilePath = sysArgs[0]
        if len(sysArgs) > 1:
            sent_model_path = sysArgs[1]

        if not os.path.exists(inputFilePath):
            inputFilePath = os.path.join(INPUT_PATH, inputFilePath)

    if not os.path.exists(inputFilePath):
        print "Input file does not exist. Terminating..."

    else:
        print "Using input file: " + inputFilePath
        inputFileFullName = os.path.split(inputFilePath)[1]
        inputFileName = os.path.splitext(inputFileFullName)[0]

        outputPath = os.path.join(OUTPUT_PATH, inputFileName)

        goldStandardPath = INPUT_PATH + inputFileName + '_aspects_list.ser'

        AAS = AspectAnalysisSystem(inputFilePath, outputPath,
                                   goldStandardPath, jobs=20,
                                   sent_model_path=sent_model_path)
        AAS.run()
