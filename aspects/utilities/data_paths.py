from pathlib import Path

from typing import Union


class ExperimentPaths:
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path], experiment_name: str = ''):
        self.input = Path(input_path)
        self.experiment_name = experiment_name

        self.output_path = Path(output_path)
        self.experiment_path = self.output_path / self.experiment_name
        self.experiment_path.mkdir(exist_ok=True, parents=True)

        self.discourse_trees_df = self.output_path / 'discourse_trees_df.pkl'

        self.aspect_sentiments = self.experiment_path / 'aspect_sentiments.pkl'
        self.aspect_to_aspect_graph = self.experiment_path / 'aspect_2_aspect_graph.pkl'
        self.conceptnet_hierarchy_neighborhood = self.experiment_path / 'conceptnet_hierarchy_neighborhood.pkl'
        self.aspect_hierarchical_tree = self.experiment_path / 'aspect_hierarchical_tree.pkl'
        self.aspect_hierarchical_tree_netwulf_config = (
                self.experiment_path / 'aspect_hierarchical_tree_netwulf_config.pkl')
        self.aspect_to_aspect_graph_flatten = self.experiment_path / 'aspect_2_aspect_graph_flatten.pkl'
        self.aspects_weighted_page_ranks = self.experiment_path / 'aspects_weighted_page_ranks.pkl'

        self.aspects_poincare_embeddings = self.experiment_path / 'aspects_poincare_embeddings'
