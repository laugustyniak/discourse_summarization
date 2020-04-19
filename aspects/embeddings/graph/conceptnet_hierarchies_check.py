from collections import defaultdict
from typing import NamedTuple, List, Set

import numpy as np
from graph_tool.stats import remove_self_loops
from graph_tool.topology import shortest_distance
from more_itertools import flatten
from tqdm import tqdm

from aspects.data.conceptnet.utils import load_english_hierarchical_graph
from aspects.data_io import serializer
from aspects.graph.convert import networkx_2_graph_tool
from aspects.utilities import settings
from aspects.utilities.data_paths import ExperimentPaths

GRAPH_TOOL_SHORTEST_PATHS_0_VALUE = 2 * 10 ^ 6


class AspectNeighborhood(NamedTuple):
    name: str
    rank: int
    neighbors_names: List[str]
    neighbors_path_lens: List[int]
    neighbors_cn_path_lens: List[int]
    aspects_not_in_conceptnet: List[int]
    cn_hierarchy_confirmed: List[bool]


def replace_zero_len_paths(shortest_paths: np.array, replaced_value: int = 0) -> np.array:
    return np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, replaced_value, shortest_paths)


def prepare_conceptnet_graph(relation_types: Set[str]):
    g = load_english_hierarchical_graph()
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
        e_hierarchical_relation_filter[edge] = edge_relation in relation_types
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
def main(max_rank: int):
    conceptnet_graph, vertices_conceptnet = prepare_conceptnet()
    aspect_graph, experiment_paths = prepare_aspect_graph()

    aspect_graph, conceptnet_graph = intersected_nodes(
        g1=aspect_graph,
        g2=conceptnet_graph,
        filter_graphs_to_intersected_vertices=False,
        property_name='aspect_name'
    )

    vertices_aspect_vertex_to_name = dict(zip(aspect_graph.vertices(), aspect_graph.vertex_properties['aspect_name']))
    vertices_name_to_aspect_vertex = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))

    if experiment_paths.conceptnet_hierarchy_neighborhood.exists():
        shortest_paths = serializer.load(experiment_paths.conceptnet_hierarchy_neighborhood)
    else:
        shortest_paths = defaultdict(list)

    for v_aspect_name in tqdm(list(aspect_graph.vertex_properties['aspect_name']), desc='Iterate over seed aspects...'):
        if v_aspect_name not in vertices_conceptnet:
            print(f'Aspect :{v_aspect_name} not in ConceptNet')
            continue
        if v_aspect_name in shortest_paths:
            print(f'Neighborhood already calculated for: {v_aspect_name}')
        else:
            v_aspect = vertices_name_to_aspect_vertex[v_aspect_name]
            for rank in range(1, max_rank + 1):
                if rank == 1:
                    neighbors = list(set(v_aspect.out_neighbors()))
                else:
                    old_neighbors = set(neighbors)
                    neighbors = list(set(flatten(v.out_neighbors() for v in old_neighbors)).difference(old_neighbors))

                all_neighbors_names = [vertices_aspect_vertex_to_name[neighbor] for neighbor in neighbors]

                # filter out aspects not present in concepnet
                neighbors_names = [
                    neighbor_name
                    for neighbor_name in all_neighbors_names
                    if neighbor_name in vertices_conceptnet
                ]

                vertices_cn_neighbors_from_aspect_graph = [
                    vertices_conceptnet[neighbor_name]
                    for neighbor_name in neighbors_names
                ]

                shortest_distances = list(shortest_distance(
                    g=conceptnet_graph,
                    source=vertices_conceptnet[v_aspect_name],
                    target=vertices_cn_neighbors_from_aspect_graph,
                    directed=True,
                ))

                neighbors_names = [
                    n
                    for idx, n
                    in enumerate(neighbors_names)
                    if shortest_distances[idx] < GRAPH_TOOL_SHORTEST_PATHS_0_VALUE
                ]
                shortest_distances = [sd for sd in shortest_distances if sd < GRAPH_TOOL_SHORTEST_PATHS_0_VALUE]

                shortest_paths[v_aspect_name].append(AspectNeighborhood(
                    name=v_aspect_name,
                    rank=rank,
                    neighbors_names=neighbors_names,
                    neighbors_path_lens=[rank for _ in neighbors_names],
                    neighbors_cn_path_lens=shortest_distances,
                    aspects_not_in_conceptnet=[aspect for aspect in all_neighbors_names if aspect in neighbors_names],
                    cn_hierarchy_confirmed=[]
                ))

                print(f'Aspect: {v_aspect_name} processed!')

            serializer.save(shortest_paths, experiment_paths.conceptnet_hierarchy_neighborhood)


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
