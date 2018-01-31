from os.path import exists, join

from os import makedirs

sample_tree_177 = '../../../data/sample_trees/177'
sample_tree_189 = '../../../data/sample_trees/189'


class IOPaths(object):
    def __init__(self, input_path, output_path):
        self.ensure_path_exist(output_path)

        self.paths = {}
        self.input = input_path

        self.extracted_docs = join(output_path, 'extracted_documents')
        self.ensure_path_exist(self.extracted_docs)

        self.extracted_docs_ids = join(output_path,
                                       'extracted_documents_ids')
        self.ensure_path_exist(self.extracted_docs_ids)

        self.extracted_docs_metadata = join(output_path,
                                            'extracted_documents_metadata')
        self.ensure_path_exist(self.extracted_docs_metadata)

        self.docs_info = join(output_path, 'documents_info')

        self.edu_trees = join(output_path, 'edu_trees_dir')
        self.ensure_path_exist(self.edu_trees)

        self.link_trees = join(output_path, 'link_trees_dir')
        self.ensure_path_exist(self.link_trees)

        self.raw_edu_list = join(output_path, 'raw_edu_list')
        self.sentiment_filtered_edus = join(output_path,
                                            'sentiment_filtered_edus')
        self.aspects_per_edu = join(output_path, 'aspects_per_edu')
        self.edu_dependency_rules = join(output_path,
                                         'edu_dependency_rules')

        self.aspects_graph = join(output_path, 'aspects_graph')
        self.aspects_graph_gpkl = join(output_path, 'aspects_graph.gpkl')
        self.aspects_graph_gexf = join(output_path, 'aspects_graph.gexf')
        self.aspects_importance = join(output_path, 'aspects_importance')

        self.final_docs_info = join(output_path,
                                    'final_documents_info')

    def ensure_path_exist(self, path):
        if not exists(path):
            makedirs(path)
