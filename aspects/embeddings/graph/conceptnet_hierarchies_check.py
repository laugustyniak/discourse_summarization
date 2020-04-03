from typing import NamedTuple, List

import numpy as np
from graph_tool import Vertex, Graph
from graph_tool.draw import graph_draw
from graph_tool.stats import remove_self_loops, linspace
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

HIERARCHICAL_RELATIONS = {'LocatedNear', 'HasA', 'MadeOf', 'PartOf', 'IsA'}


def replace_zero_len_paths(shortest_paths: np.array, replaced_value: int = 0) -> np.array:
    return np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, replaced_value, shortest_paths)


def prepare_conceptnet_graph():
    g = load_english_graph()
    remove_self_loops(g)
    g.reindex_edges()

    # filter relations
    e_hierarchical_relation_filter = g.new_edge_property('bool')
    relations = list(g.properties[('e', 'relation')])
    for edge, edge_relation in tqdm(
            zip(
                g.edges(),
                relations
            ),
            desc='Edge filtering...',
            total=len(relations)
    ):
        e_hierarchical_relation_filter[edge] = edge_relation in HIERARCHICAL_RELATIONS
    g.set_edge_filter(e_hierarchical_relation_filter)

    vertices = dict(zip(g.vertex_properties['aspect_name'], g.vertices()))

    return g, vertices


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
    print(f'Pre-filter graph stats: {g}')
    v_connected = g.new_vertex_property('bool')
    for v in g.vertices():
        v_connected[v] = v.in_degree() > 0 or v.out_degree() > 0
    g.set_vertex_filter(v_connected)
    print(f'Post-filter graph stats: {g}')
    return Graph(g, prune=True, directed=g.is_directed())


if __name__ == '__main__':
    conceptnet_graph, vertices_conceptnet = prepare_conceptnet_graph()

    experiment_paths = ExperimentPaths(
        input_path='',
        output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs',
        experiment_name='our'
    )
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)
    aspect_graph.reindex_edges()

    aspect_graph, conceptnet_graph = intersected_nodes(
        g1=aspect_graph,
        g2=conceptnet_graph,
        filter_graphs_to_intersected_vertices=False,
        property_name='aspect_name'
    )

    conceptnet_graph = remove_not_connected_vertices(conceptnet_graph)

    graph_draw(
        conceptnet_graph,
        vertex_font_size=12,
        output_size=(2000, 2000),
        output="conceptnet-intersected-aspects.png",
        bg_color='white',
        vertex_text=conceptnet_graph.vp.aspect_name
    )

    vertices_aspect_graph = dict(zip(aspect_graph.vertices(), aspect_graph.vertex_properties['aspect_name']))
    vertices_aspect_graph_reverted = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))

    shortest_paths = {}
    for v in tqdm(aspect_graph.vertices(), desc='Iterate over vertices...'):

        v_name = vertices_aspect_graph[v]
        if v_name in SEED_ASPECTS:
            rank = 1
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
                    shortest_paths_aspects.append(rank)

            shortest_paths[v_name] = AspectNeighborhood(
                vertex=v,
                name=v_name,
                rank=rank,
                neighbors_names=neighbors_names,
                neighbors_path_lens=shortest_paths_aspects,
                neighbors_cn_path_lens=list(replace_zero_len_paths(np.array(shortest_paths_conceptnet)))
            )

    serializer.save(shortest_paths, experiment_paths.conceptnet_hierarchy_neighborhood)

    d = g.degree_property_map("out", weight)  # weight is an edge property map
    bins = linspace(d.a.min(), d.a.max(), 40)  # linear bins
    h = vertex_hist(g, d, bins)

    # shortest_paths = shortest_distance(aspect_graph)
    # shortest_paths = np.array(list(shortest_paths))
    # shortest_paths = np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, 0, shortest_paths)

    print(aspect_graph)
