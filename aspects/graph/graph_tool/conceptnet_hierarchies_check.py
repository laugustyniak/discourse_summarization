import logging
from pathlib import Path
from typing import Union, Tuple, Dict

import graph_tool as gt
import numpy as np
import pandas as pd
from graph_tool import Graph
from graph_tool.stats import remove_self_loops
from graph_tool.topology import shortest_distance
from tqdm import tqdm

from aspects.data_io import serializer
from aspects.graph.convert import networkx_2_graph_tool
from aspects.graph.graph_tool.utils import GRAPH_TOOL_SHORTEST_PATHS_0_VALUE
from aspects.utilities.data_paths import ExperimentPaths

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def replace_zero_len_paths(shortest_paths: np.array, replaced_value: int = 0) -> np.array:
    return np.where(shortest_paths > GRAPH_TOOL_SHORTEST_PATHS_0_VALUE, replaced_value, shortest_paths)


def intersected_nodes(g1, g2, filter_graphs_to_intersected_vertices: bool = False, property_name: str = 'aspect_name'):
    logger.info(f'g1: {g1}')
    logger.info(f'g2: {g2}')
    g1_nodes = set(g1.vp[property_name])
    g2_nodes = set(g2.vp[property_name])
    g1_and_g2 = g1_nodes.intersection(g2_nodes)
    g1_not_in_g2 = g1_nodes.difference(g2_nodes)
    logger.info(
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
    logger.info(f'Pre-edge-purge graph stats: {g}')
    g.purge_edges()
    logger.info(f'Pre-vertices-purge graph stats: {g}')
    g.purge_vertices(in_place=True)
    logger.info(f'Pre-filter graph stats: {g}')
    v_connected = g.new_vertex_property('bool')
    for v in tqdm(g.vertices(), desc='Vertices filtering...', total=g.num_vertices()):
        v_connected[v] = bool(v.in_degree() or v.out_degree())
    g.set_vertex_filter(v_connected)
    g.purge_vertices(in_place=False)
    logger.info(f'Post-filter graph stats: {g}')
    return g


def prepare_hierarchies_neighborhood(experiments_path: ExperimentPaths, conceptnet_graph_path: Union[str, Path]):
    logger.info('Prepare graphs')
    conceptnet_graph, vertices_conceptnet = prepare_conceptnet(conceptnet_graph_path)
    aspect_graph, experiment_paths = prepare_aspect_graph(experiments_path)

    aspect_graph_intersected = Graph(aspect_graph)
    conceptnet_graph_intersected = Graph(conceptnet_graph)

    aspect_graph_intersected, conceptnet_graph_intersected = intersected_nodes(
        g1=aspect_graph_intersected,
        g2=conceptnet_graph_intersected,
        filter_graphs_to_intersected_vertices=True,
        property_name='aspect_name'
    )
    aspect_names_intersected = list(aspect_graph_intersected.vertex_properties['aspect_name'])

    vertices_name_to_aspect_vertex = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))
    aspect_graph_vertices_intersected = [vertices_name_to_aspect_vertex[a] for a in aspect_names_intersected]
    shortest_distances_aspect_graph = np.array([
        shortest_distance(g=aspect_graph, source=v, target=aspect_graph_vertices_intersected, directed=True)
        for v in tqdm(aspect_graph_vertices_intersected, desc='Aspect graph shortest paths...')
    ])

    conceptnet_vertices_intersected = [vertices_conceptnet[a] for a in aspect_names_intersected]

    assert conceptnet_vertices_intersected == aspect_graph_vertices_intersected, 'Wrong sequence of vertices in both graphs'

    shortest_distances_conceptnet = np.array([
        shortest_distance(g=conceptnet_graph, source=v, target=conceptnet_vertices_intersected, directed=True)
        for v in tqdm(conceptnet_vertices_intersected, desc='Conceptnet shortest paths...')
    ])

    pairs = []
    for aspect_1_idx, aspect_1 in tqdm(enumerate(aspect_names_intersected), desc='Pairs distances...'):
        for aspect_2_idx, aspect_2 in enumerate(aspect_names_intersected):
            pairs.append((
                aspect_1,
                aspect_2,
                shortest_distances_aspect_graph[aspect_1_idx][aspect_2_idx],
                shortest_distances_conceptnet[aspect_1_idx][aspect_2_idx],
            ))

    pairs_df = pd.DataFrame(
        pairs,
        columns=['aspect_1', 'aspect_2', 'shortest_distance_aspect_graph', 'shortest_distance_conceptnet']
    )

    logger.info('Dump DataFrame with pairs')
    pairs_df.to_pickle(experiment_paths.conceptnet_hierarchy_neighborhood)
    logger.info(f'DataFrame with pairs dumped in: {experiment_paths.conceptnet_hierarchy_neighborhood.as_posix()}')


def prepare_aspect_graph(experiment_paths: ExperimentPaths) -> Tuple[Graph, ExperimentPaths]:
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)
    aspect_graph.reindex_edges()
    # revert edges from S -> N to N -> S
    aspect_graph.set_reversed(is_reversed=True)
    return Graph(aspect_graph), experiment_paths


def prepare_conceptnet(graph_path: Union[str, Path]) -> Tuple[Graph, Dict[str, gt.Vertex]]:
    conceptnet_graph = gt.load_graph(str(graph_path))
    remove_self_loops(conceptnet_graph)
    conceptnet_graph.reindex_edges()
    vertices_conceptnet = dict(zip(conceptnet_graph.vertex_properties['aspect_name'], conceptnet_graph.vertices()))
    return Graph(conceptnet_graph), vertices_conceptnet
