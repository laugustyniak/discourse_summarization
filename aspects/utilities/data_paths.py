from os import makedirs
from os.path import exists, join

from pathlib import Path


# TODO: remove class + variables by upper case
class IOPaths(object):
    def __init__(self, input_path, output_path, suffix=''):
        self.ensure_path_exist(output_path)

        self.paths = {}
        self.input = input_path

        self.extracted_docs = join(output_path, 'extracted_documents')
        self.ensure_path_exist(self.extracted_docs)

        self.extracted_docs_ids = join(output_path, 'extracted_documents_ids')
        self.ensure_path_exist(self.extracted_docs_ids)

        self.extracted_docs_metadata = join(output_path, 'extracted_documents_metadata')
        self.ensure_path_exist(self.extracted_docs_metadata)

        self.docs_info = join(output_path, 'documents_info')

        self.edu_trees = join(output_path, 'edu_trees_dir')
        self.ensure_path_exist(self.edu_trees)

        self.discourse_trees_df = Path(output_path) / 'link_trees_df.pkl'

        self.raw_edus = join(output_path, 'raw_edu_list')
        self.sentiment_filtered_edus = join(output_path, 'sentiment_filtered_edus')
        self.aspects_per_edu = join(output_path, 'aspects_per_edu')
        self.edu_dependency_rules = join(output_path, 'edu_dependency_rules')

        self.aspects_graph = join(output_path, self._add_suffix('aspects_graph', suffix))
        self.aspects_graph_gpkl = join(output_path, self._add_suffix('aspects_graph.gpkl', suffix))
        self.aspects_graph_gexf = join(output_path, self._add_suffix('aspects_graph.gexf', suffix))
        self.aspects_page_ranks = join(output_path, self._add_suffix('aspects_page_ranks', suffix))

        self.final_docs_info = join(output_path, self._add_suffix('final_documents_info', suffix))

    def ensure_path_exist(self, path):
        if not exists(path):
            makedirs(path)

    def _add_suffix(self, name, suffix):
        name = Path(name)
        return '{}_{}{}'.format(name.stem, suffix, name.suffix) if suffix else name.name
