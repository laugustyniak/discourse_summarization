import logging
from itertools import product

import numpy as np
import pandas as pd
from graph_tool.stats import remove_self_loops
from graph_tool.topology import shortest_distance
from tqdm import tqdm

from aspects.data.conceptnet.utils import load_english_hierarchical_graph
from aspects.data_io import serializer
from aspects.graph.convert import networkx_2_graph_tool
from aspects.utilities import settings
from aspects.utilities.data_paths import ExperimentPaths

GRAPH_TOOL_SHORTEST_PATHS_0_VALUE = 2 * 10 ^ 6

loger = logging.getLogger()
loger.setLevel(logging.DEBUG)


def replace_zero_len_paths(shortest_paths: np.array, replaced_value: int = 0) -> np.array:
    return np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, replaced_value, shortest_paths)





def intersected_nodes(g1, g2, filter_graphs_to_intersected_vertices: bool = False, property_name: str = 'aspect_name'):
    print(f'g1: {g1}')
    print(f'g2: {g2}')
    g1_nodes = set(g1.vp[property_name])
    g2_nodes = set(g2.vp[property_name])
    g1_and_g2 = g1_nodes.intersection(g2_nodes)
    g1_not_in_g2 = g1_nodes.difference(g2_nodes)
    print(
        f'g1 nodes: #{len(g1_nodes)}\n'
        f'g2 nodes: #{len(g2_nodes)}\n'
        f'g1 and g2 nodes: #{len(g1_and_g2)}\n'
        f'g1 not in g2 nodes: #{len(g1_not_in_g2)}\n'
    )

    if filter_graphs_to_intersected_vertices:
        v2_intersected = g2.new_vertex_property('bool')
        for v, v_name in tqdm(zip(g2.vertices(), list(g2.vp[property_name])), desc='Vertices filtering...'):
            v2_intersected[v] = v_name in g1_and_g2
        g2.set_vertex_filter(v2_intersected)

        v1_intersected = g1.new_vertex_property('bool')
        for v, v_name in tqdm(zip(g1.vertices(), list(g1.vp[property_name])), desc='Vertices filtering...'):
            v1_intersected[v] = v_name in g1_and_g2
        g1.set_vertex_filter(v1_intersected)

    return g1, g2


def remove_not_connected_vertices(g):
    print(f'Pre-edge-purge graph stats: {g}')
    g.purge_edges()
    print(f'Pre-vertices-purge graph stats: {g}')
    g.purge_vertices(in_place=True)
    print(f'Pre-filter graph stats: {g}')
    v_connected = g.new_vertex_property('bool')
    for v in tqdm(g.vertices(), desc='Vertices filtering...', total=g.num_vertices()):
        v_connected[v] = bool(v.in_degree() or v.out_degree())
    g.set_vertex_filter(v_connected)
    g.purge_vertices(in_place=False)
    print(f'Post-filter graph stats: {g}')
    return g


# TODO: add click
def main():
    loger.info('Prepare graphs')
    conceptnet_graph, vertices_conceptnet = prepare_conceptnet()
    aspect_graph, experiment_paths = prepare_aspect_graph()

    # some graphs stats
    aspect_graph, conceptnet_graph = intersected_nodes(
        g1=aspect_graph,
        g2=conceptnet_graph,
        filter_graphs_to_intersected_vertices=False,
        property_name='aspect_name'
    )

    vertices_name_to_aspect_vertex = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))

    loger.info('Calculate shortest paths: aspect graph')
    shortest_distances_aspect_graph = shortest_distance(g=aspect_graph, directed=True)
    loger.info('Calculate shortest paths: conceptnet graph')
    shortest_distances_conceptnet = shortest_distance(g=conceptnet_graph, directed=True)
    loger.info('Calculate shortest paths: done')

    aspect_graph_vertices = list(aspect_graph.vertex_properties['aspect_name'])

    pairs = []
    for aspect_1, aspect_2 in tqdm(
            product(aspect_graph_vertices, aspect_graph_vertices),
            total=(aspect_graph.num_vertices() * aspect_graph.num_vertices())
    ):
        pairs.append((
            aspect_1,
            aspect_2,
            shortest_distances_aspect_graph[
                int(vertices_name_to_aspect_vertex[aspect_1])][
                int(vertices_name_to_aspect_vertex[aspect_2])
            ],
            shortest_distances_aspect_graph[
                int(shortest_distances_conceptnet[aspect_1])][
                int(shortest_distances_conceptnet[aspect_2])
            ],
        ))

    pairs_df = pd.DataFrame(
        pairs,
        columns=['aspect_1', 'aspect_2', 'shortest_distance_aspect_graph', 'shortest_distance_conceptnet']
    )

    loger.info('Dump DataFrame with pairs')
    pairs_df.to_pickle(experiment_paths.conceptnet_hierarchy_neighborhood)
    loger.info('DataFrame with pairs dumped')


def prepare_aspect_graph():
    experiment_paths = ExperimentPaths(
        input_path='',
        output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs',
        # output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Apps_for_Android',
        # output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Amazon_Instant_Video',
        experiment_name='our'
    )
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)
    aspect_graph.reindex_edges()
    # revert edges from S -> N to N -> S
    aspect_graph.set_reversed(is_reversed=True)
    return aspect_graph, experiment_paths


def prepare_conceptnet():
    conceptnet_graph = load_english_hierarchical_graph()
    remove_self_loops(conceptnet_graph)
    conceptnet_graph.reindex_edges()
    vertices_conceptnet = dict(zip(conceptnet_graph.vertex_properties['aspect_name'], conceptnet_graph.vertices()))
    return conceptnet_graph, vertices_conceptnet


if __name__ == '__main__':
    main(3)
