from collections import defaultdict
from typing import NamedTuple, List, Set

import numpy as np
from graph_tool import Vertex, GraphView
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
    vertex: Vertex
    name: str
    rank: int
    neighbors_names: List[str]
    neighbors_path_lens: List[int]
    neighbors_cn_path_lens: List[int]
    aspects_not_in_conceptnet: List[int]
    cn_hierarchy_confirmed: List[bool]


# TODO: param
SEED_ASPECTS = ['bluetooth', 'phone', 'battery', 'price']


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


if __name__ == '__main__':
    conceptnet_graph = load_english_hierarchical_graph()
    remove_self_loops(conceptnet_graph)
    conceptnet_graph.reindex_edges()
    vertices_conceptnet = dict(zip(conceptnet_graph.vertex_properties['aspect_name'], conceptnet_graph.vertices()))

    experiment_paths = ExperimentPaths(
        input_path='',
        output_path=settings.DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs',
        experiment_name='our'
    )
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)
    aspect_graph.reindex_edges()

    # revert edges fomr S->N to N-> S
    aspect_graph.set_reversed(is_reversed=True)

    aspect_graph, conceptnet_graph = intersected_nodes(
        g1=aspect_graph,
        g2=conceptnet_graph,
        filter_graphs_to_intersected_vertices=False,
        property_name='aspect_name'
    )

    # graph_draw(
    #     conceptnet_graph,
    #     vertex_font_size=12,
    #     output_size=(2000, 2000),
    #     output="conceptnet-intersected-aspects.png",
    #     bg_color='white',
    #     vertex_text=conceptnet_graph.vp.aspect_name
    # )

    vertices_aspect_vertex_to_name = dict(zip(aspect_graph.vertices(), aspect_graph.vertex_properties['aspect_name']))
    vertices_name_to_aspect_vertex = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))

    shortest_paths = defaultdict(dict)
    # TODO: param
    max_rank = 3

    # TODO: param
    u = GraphView(aspect_graph, vfilt=lambda v: v.out_degree() > 10)
    for v_aspect_name in tqdm(list(u.vertex_properties['aspect_name'])[:20], desc='Iterate over seed aspects...'):
        v_aspect = vertices_name_to_aspect_vertex[v_aspect_name]
        for rank in range(1, max_rank + 1):
            if rank == 1:
                neighbors = list(set(v_aspect.out_neighbors()))
            else:
                old_neighbors = set(neighbors)
                neighbors = list(set(flatten(v.out_neighbors() for v in old_neighbors)).difference(old_neighbors))

            neighbors_names = [vertices_aspect_vertex_to_name[neighbor] for neighbor in neighbors]
            shortest_paths_conceptnet = []
            shortest_paths_aspects = []
            aspects_not_in_conceptnet = []
            cn_hierarchy_confirmed = []
            for neighbor_name in neighbors_names:
                try:
                    if neighbor_name in vertices_conceptnet:
                        sh_dist = shortest_distance(
                            g=conceptnet_graph,
                            source=vertices_conceptnet[v_aspect_name],
                            target=vertices_conceptnet[neighbor_name],
                            directed=True,
                        )
                        sh_dist_reversed = shortest_distance(
                            g=conceptnet_graph,
                            source=vertices_conceptnet[neighbor_name],
                            target=vertices_conceptnet[v_aspect_name],
                            directed=True,
                        )
                        shortest_paths_conceptnet.append(sh_dist)
                        cn_hierarchy_confirmed.append(sh_dist < sh_dist_reversed)
                        shortest_paths_aspects.append(rank)
                    else:
                        aspects_not_in_conceptnet.append(neighbor_name)
                except KeyError:
                    pass

            shortest_paths[v_aspect_name][str(rank)] = AspectNeighborhood(
                vertex=v_aspect,
                name=v_aspect_name,
                rank=rank,
                neighbors_names=neighbors_names,
                neighbors_path_lens=shortest_paths_aspects,
                neighbors_cn_path_lens=list(replace_zero_len_paths(np.array(shortest_paths_conceptnet))),
                aspects_not_in_conceptnet=aspects_not_in_conceptnet,
                cn_hierarchy_confirmed=cn_hierarchy_confirmed
            )

    serializer.save(shortest_paths, experiment_paths.conceptnet_hierarchy_neighborhood)

    # shortest_paths = shortest_distance(aspect_graph)
    # shortest_paths = np.array(list(shortest_paths))
    # shortest_paths = np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, 0, shortest_paths)
