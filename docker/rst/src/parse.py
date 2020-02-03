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
        if len(doc.edus) > 1:
            for idx, edu_tokens in enumerate(doc.edus):
                # sometimes RST parser duplicates last three chars and adds a new token in edu, remove it
                if edu_tokens[-2].endswith(edu_tokens[-1]):
                    edu_tokens = edu_tokens[:-1]
                pt.__setitem__(pt.leaf_treeposition(idx), '_!%s!_' % ' '.join(edu_tokens))
        else:
            pt = pt[0]
            pt.__setitem__(pt.leaf_treeposition(0), '_!%s!_' % ' '.join(doc.edus[0]))

        return str(pt)
