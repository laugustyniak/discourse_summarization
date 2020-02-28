from pathlib import Path

from typing import Union


class IOPaths:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path], suffix: str = None):
        if suffix is None:
            self.suffix = ''

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.input = input_path

        self.discourse_trees_df = self.output_path / 'discourse_trees_df.pkl'

        self.aspects_graph = self.output_path / self._add_suffix('aspects_graph.pkl')
        self.spanning_tree = self.output_path / self._add_suffix('spanning_tree.pkl')
        self.graph_flatten = self.output_path / self._add_suffix('graph_flatten.pkl')
        self.aspect_sentiments = self.output_path / self._add_suffix('aspect_sentiments.pkl')
        self.aspects_weighted_page_ranks = self.output_path / self._add_suffix('aspects_page_ranks')

        self.final_docs_info = self.output_path / self._add_suffix('final_documents_info')

    def _add_suffix(self, name: str):
        name = Path(name)
        return '{}_{}{}'.format(name.stem, self.suffix, name.suffix) if self.suffix else name.name
