import os.path
import traceback

from document.doc import Document
from prep.preprocesser import Preprocesser
from segmenters.crf_segmenter import CRFSegmenter
from treebuilder.build_tree_CRF import CRFTreeBuilder
from trees.parse_tree import ParseTree


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

        if self.segmenter is not None:
            self.segmenter.unload()

        if self.treebuilder is not None:
            self.treebuilder.unload()

    def parse(self, text):
        assert len(text) > 0, "Text can not be empty"
        doc = Document()
        doc.preprocess(text, self.preprocesser)

        if not doc.segmented:
            self.segmenter.segment(doc)

        ''' Step 2: build text-level discourse tree '''
        if self.skip_parsing:
            sent_out_list = []
            for sentence in doc.sentences:
                sent_id = sentence.sent_id
                edu_segmentation = doc.edu_word_segmentation[sent_id]
                i = 0
                sent_out = []
                for (j, token) in enumerate(sentence.tokens):
                    sent_out.append(token.word)
                    if j < len(sentence.tokens) - 1 and j == edu_segmentation[i][1] - 1:
                        sent_out.append('EDU_BREAK')
                        i += 1
                sent_out_list.append(' '.join(sent_out))
            return sent_out_list
        else:
            pt = self.treebuilder.build_tree(doc)
            if pt is None:
                if self.treebuilder is not None:
                    self.treebuilder.unload()
                return -1

            # Unescape the parse tree
            if pt:
                doc.discourse_tree = pt
                if type(pt) is list:
                    if isinstance(pt[0], ParseTree):
                        pt = pt[0]
                for i in range(len(doc.edus)):
                    pt.__setitem__(pt.leaf_treeposition(i), '_!%s!_' % ' '.join(doc.edus[i]))
                return pt
