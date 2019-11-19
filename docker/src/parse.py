import os.path
import traceback

from document.doc import Document
from prep.preprocesser import Preprocesser
from segmenters.crf_segmenter import CRFSegmenter
from treebuilder.build_tree_CRF import CRFTreeBuilder


class DiscourseParser(object):
    def __init__(self, output_dir=None, verbose=False,
                 skip_parsing=False, global_features=False,
                 save_preprocessed_doc=False, preprocesser=None):

        self.output_dir = os.path.join(output_dir if output_dir is not None else '')
        self.feature_sets = 'gCRF'

        self.verbose = verbose
        self.skip_parsing = skip_parsing
        self.global_features = global_features
        self.save_preprocessed_doc = save_preprocessed_doc

        if preprocesser is not None:
            self.preprocesser = preprocesser

        try:
            self.preprocesser = Preprocesser()
        except Exception, e:
            print "*** Loading Preprocessing module failed..."
            print traceback.print_exc()

            raise e
        try:
            self.segmenter = CRFSegmenter(_name=self.feature_sets,
                                          verbose=self.verbose,
                                          global_features=self.global_features)
        except Exception, e:
            print "*** Loading Segmentation module failed..."
            print traceback.print_exc()

            raise e

        try:
            if not self.skip_parsing:
                self.treebuilder = CRFTreeBuilder(_name=self.feature_sets,
                                                  verbose=self.verbose)
            else:
                self.treebuilder = None
        except Exception, e:
            print "*** Loading Tree-building module failed..."
            print traceback.print_exc()
            raise e

    def unload(self):
        if self.preprocesser is not None:
            self.preprocesser.unload()

        if not self.segmenter is None:
            self.segmenter.unload()

        if not self.treebuilder is None:
            self.treebuilder.unload()

    def parse(self, text=''):
        doc = Document()
        doc.preprocess(text, self.preprocesser)

        if not doc.segmented:
            self.segmenter.segment(doc)

        pt = self.treebuilder.build_tree(doc)
        doc.discourse_tree = pt
        for i in range(len(doc.edus)):
            pt.__setitem__(pt.leaf_treeposition(i), ' '.join(doc.edus[i]))
        return str(pt)
