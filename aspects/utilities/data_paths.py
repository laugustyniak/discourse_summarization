from pathlib import Path

from typing import Union


class ExperimentPaths:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        self.input = Path(input_path)

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)

        self.discourse_trees_df = self.output_path / 'discourse_trees_df.pkl'
        self.aspect_to_aspect_graph = self.output_path / 'aspect_2_aspect_graph.pkl'
        self.aspect_hierarchical_tree = self.output_path / 'aspect_hierarchical_tree.pkl'
        self.aspect_to_aspect_graph_flatten = self.output_path / 'aspect_2_aspect_graph_flatten.pkl'
        self.aspect_sentiments = self.output_path / 'aspect_sentiments.pkl'
        self.aspects_weighted_page_ranks = self.output_path / 'aspects_weighted_page_ranks.pkl'
