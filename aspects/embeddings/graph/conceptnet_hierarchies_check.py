from typing import NamedTuple, List

import numpy as np
from graph_tool import Vertex
from graph_tool.draw import graph_draw
from graph_tool.stats import remove_self_loops
from graph_tool.topology import shortest_distance
from tqdm import tqdm

from aspects.data.conceptnet.utils import load_english_graph
from aspects.data_io import serializer
from aspects.graph.convert import networkx_2_graph_tool
from aspects.utilities import settings
from aspects.utilities.data_paths import ExperimentPaths

GRAPH_TOOL_SHORTEST_PATHS_0_VALUE = 2 * 10 ^ 6


class AspectNeighborhood(NamedTuple):
    vertex: Vertex
    name: str
    rank: int
    neighbors_names: List[str]
    neighbors_path_lens: List[int]
    neighbors_cn_path_lens: List[int]


SEED_ASPECTS = ['phone', 'battery', 'price']


def replace_zero_len_paths(shortest_paths: np.array, replaced_value: int = 0) -> np.array:
    return np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, replaced_value, shortest_paths)


if __name__ == '__main__':
    conceptnet_graph = load_english_graph()
    remove_self_loops(conceptnet_graph)
    vertices_conceptnet = dict(zip(conceptnet_graph.vertex_properties['aspect_name'], conceptnet_graph.vertices()))

    experiment_paths = ExperimentPaths(
        input_path='',
        output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs',
        experiment_name='our'
    )
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)

    vertices_aspect_graph = dict(zip(aspect_graph.vertices(), aspect_graph.vertex_properties['aspect_name']))
    vertices_aspect_graph_reverted = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))

    shortest_paths = {}
    for v in tqdm(aspect_graph.vertices(), desc='Iterate over vertices...'):

        v_name = vertices_aspect_graph[v]
        if v_name in SEED_ASPECTS:
            neighbors = list(set(v.out_neighbors()))
            neighbors_names = [vertices_aspect_graph[neighbor] for neighbor in neighbors]

            shortest_paths_conceptnet = []
            shortest_paths_aspects = []
            for v_neighbor in neighbors:
                neighbor_name = vertices_aspect_graph[v_neighbor]
                if neighbor_name in vertices_conceptnet:
                    shortest_paths_conceptnet.append(
                        shortest_distance(
                            g=conceptnet_graph,
                            source=vertices_conceptnet[v_name],
                            target=vertices_conceptnet[neighbor_name]
                        )
                    )
                    shortest_paths_aspects.append(1)

            shortest_paths[v_name] = AspectNeighborhood(
                vertex=v,
                name=v_name,
                rank=1,
                neighbors_names=neighbors_names,
                neighbors_path_lens=shortest_paths_aspects,
                neighbors_cn_path_lens=list(replace_zero_len_paths(np.array(shortest_paths_conceptnet)))
            )

    # shortest_paths = shortest_distance(aspect_graph)
    # shortest_paths = np.array(list(shortest_paths))
    # shortest_paths = np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, 0, shortest_paths)

    graph_draw(aspect_graph, vertex_font_size=18, output_size=(800, 800), output="aspect-test-graph.png")

    print(aspect_graph)
