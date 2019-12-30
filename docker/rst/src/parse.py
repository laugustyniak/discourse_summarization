import os.path

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

        self.preprocesser = Preprocesser()

        self.segmenter = CRFSegmenter(
            _name=self.feature_sets, verbose=self.verbose, global_features=self.global_features)
        if not self.skip_parsing:
            self.treebuilder = CRFTreeBuilder(_name=self.feature_sets, verbose=self.verbose)
        else:
            self.treebuilder = None

    # TODO: remove in future
    def unload(self):
        if self.preprocesser is not None:
            self.preprocesser.unload()

        if self.segmenter is not None:
            self.segmenter.unload()

        if self.treebuilder is not None:
            self.treebuilder.unload()

    def parse(self, text):
        doc = Document()
        doc.preprocess(text, self.preprocesser)

        if not doc.segmented:
            self.segmenter.segment(doc)

        pt = self.treebuilder.build_tree(doc)
        doc.discourse_tree = pt
        for i in range(len(doc.edus)):
            pt.__setitem__(pt.leaf_treeposition(i), '_!%s!_' % ' '.join(doc.edus[i]))
            # pt.__setitem__(pt.leaf_treeposition(i), ' '.join(doc.edus[i]))
        return str(pt)