from pathlib import Path

from typing import Union


class IOPaths:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path], suffix: str = None):
        if suffix is None:
            self.suffix = ''

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.paths = {}
        self.input = input_path

        self.discourse_trees_df = Path(output_path) / 'discourse_trees_df.pkl'

        self.aspects_graph = self.output_path / self._add_suffix('aspects_graph')
        self.aspects_graph_gpkl = self.output_path / self._add_suffix('aspects_graph.gpkl')
        self.aspects_graph_gexf = self.output_path / self._add_suffix('aspects_graph.gexf')
        self.aspects_page_ranks = self.output_path / self._add_suffix('aspects_page_ranks')

        self.final_docs_info = self.output_path / self._add_suffix('final_documents_info')

    def _add_suffix(self, name: str):
        name = Path(name)
        return '{}_{}{}'.format(name.stem, self.suffix, name.suffix) if self.suffix else name.name
