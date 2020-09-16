import logging
from functools import lru_cache
from pathlib import Path
from typing import Union, Tuple, Dict

import graph_tool as gt
import mlflow
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


def intersected_nodes(
        aspect_graph,
        conceptnet_graph,
        filter_graphs_to_intersected_vertices: bool = False,
        property_name: str = 'aspect_name'
):
    logger.info(f'aspect_graph: {aspect_graph}')
    logger.info(f'conceptnet_graph: {conceptnet_graph}')
    g1_nodes = set(aspect_graph.vp[property_name])
    g2_nodes = set(conceptnet_graph.vp[property_name])
    g1_and_g2 = g1_nodes.intersection(g2_nodes)
    mlflow.log_metric('graphs_intersected_nodes', len(g1_and_g2))
    g1_not_in_g2 = g1_nodes.difference(g2_nodes)
    mlflow.log_metric('only_in_aspect_graph_nodes', len(g1_not_in_g2))
    logger.info(
        f'aspect_graph nodes: #{len(g1_nodes)}\n'
        f'conceptnet_graph nodes: #{len(g2_nodes)}\n'
        f'aspect_graph and conceptnet_graph nodes: #{len(g1_and_g2)}\n'
        f'aspect_graph not in conceptnet_graph nodes: #{len(g1_not_in_g2)}\n'
    )

    if filter_graphs_to_intersected_vertices:
        v2_intersected = conceptnet_graph.new_vertex_property('bool')
        for v, v_name in tqdm(zip(
                conceptnet_graph.vertices(),
                list(conceptnet_graph.vp[property_name])
        ), desc='Vertices filtering...'):
            v2_intersected[v] = v_name in g1_and_g2
        conceptnet_graph.set_vertex_filter(v2_intersected)

        v1_intersected = aspect_graph.new_vertex_property('bool')
        for v, v_name in tqdm(zip(
                aspect_graph.vertices(),
                list(aspect_graph.vp[property_name])
        ), desc='Vertices filtering...'):
            v1_intersected[v] = v_name in g1_and_g2
        aspect_graph.set_vertex_filter(v1_intersected)

    return aspect_graph, conceptnet_graph


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


def prepare_hierarchies_neighborhood(
        experiments_path: ExperimentPaths, conceptnet_graph_path: Union[str, Path]
) -> pd.DataFrame:

    conceptnet_hierarchy_neighborhood_df_path = (
            experiments_path.experiment_path / f'shortest-paths-pairs-{conceptnet_graph_path.stem}-df.pkl')

    if conceptnet_hierarchy_neighborhood_df_path.exists():
        return pd.read_pickle(conceptnet_hierarchy_neighborhood_df_path.as_posix())

    logger.info('Prepare graphs')

    conceptnet_graph, vertices_conceptnet = prepare_conceptnet(conceptnet_graph_path)
    mlflow.log_metric('conceptnet_graph_nodes', conceptnet_graph.num_vertices())
    mlflow.log_metric('conceptnet_graph_edges', conceptnet_graph.num_edges())

    aspect_graph, experiment_paths = prepare_aspect_graph(experiments_path)
    mlflow.log_metric('aspect_graph_nodes', aspect_graph.num_vertices())
    mlflow.log_metric('aspect_graph_edges', aspect_graph.num_edges())

    aspect_graph_intersected = Graph(aspect_graph)
    conceptnet_graph_intersected = Graph(conceptnet_graph)

    aspect_graph_intersected, conceptnet_graph_intersected = intersected_nodes(
        aspect_graph=aspect_graph_intersected,
        conceptnet_graph=conceptnet_graph_intersected,
        # TODO: check, czy usuwajac wierzcholki nie usuwam tez krawedzi z nimi powiazanych? wtedy mam rzadszy graf i calkiem inne relacje niz na calym grafie
        filter_graphs_to_intersected_vertices=True,
        property_name='aspect_name'
    )

    mlflow.log_metric('conceptnet_graph_intersected_nodes', conceptnet_graph_intersected.num_vertices())
    mlflow.log_metric('aspect_graph_intersected_nodes', aspect_graph_intersected.num_vertices())

    mlflow.log_metric('conceptnet_graph_intersected_edges', conceptnet_graph_intersected.num_edges())
    mlflow.log_metric('aspect_graph_intersected_edges', aspect_graph_intersected.num_edges())

    aspect_names_intersected = list(aspect_graph_intersected.vertex_properties['aspect_name'])

    vertices_name_to_aspect_vertex = dict(zip(aspect_graph.vertex_properties['aspect_name'], aspect_graph.vertices()))
    aspect_graph_vertices_intersected = [vertices_name_to_aspect_vertex[a] for a in aspect_names_intersected]
    shortest_distances_aspect_graph = np.array([
        shortest_distance(g=aspect_graph, source=v, target=aspect_graph_vertices_intersected, directed=True)
        for v in tqdm(aspect_graph_vertices_intersected, desc='Aspect graph shortest paths...')
    ])

    conceptnet_vertices_intersected = [vertices_conceptnet[a] for a in aspect_names_intersected]

    logger.info(f'conceptnet_vertices_intersected len = {len(conceptnet_vertices_intersected)}')
    logger.info(f'aspect_graph_vertices_intersected len = {len(aspect_graph_vertices_intersected)}')
    assert len(conceptnet_vertices_intersected) == len(aspect_graph_vertices_intersected), 'Wrong sequence of vertices in both graphs'

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
    pairs_df.to_pickle(conceptnet_hierarchy_neighborhood_df_path.as_posix())
    mlflow.log_artifact(experiment_paths.conceptnet_hierarchy_neighborhood)
    mlflow.log_metric('conceptnet_hierarchy_neighborhood_df_len', len(pairs_df))
    logger.info(f'DataFrame with pairs dumped in: {experiment_paths.conceptnet_hierarchy_neighborhood.as_posix()}')

    return pairs_df


@lru_cache(maxsize=1)
def prepare_aspect_graph(experiment_paths: ExperimentPaths) -> Tuple[Graph, ExperimentPaths]:
    logger.info(f'Load aspect 2 aspect graph - {str(experiment_paths.aspect_to_aspect_graph)}')
    aspect_graph = serializer.load(experiment_paths.aspect_to_aspect_graph)
    aspect_graph = networkx_2_graph_tool(aspect_graph, node_name_property='aspect_name')
    remove_self_loops(aspect_graph)
    aspect_graph.reindex_edges()
    # TODO: check if the direction is correct
    # revert edges from S -> N to N -> S
    aspect_graph.set_reversed(is_reversed=True)
    return Graph(aspect_graph), experiment_paths


@lru_cache(maxsize=1)
def prepare_conceptnet(graph_path: Union[str, Path]) -> Tuple[Graph, Dict[str, gt.Vertex]]:
    logger.info(f'Load conceptnet graph - {str(graph_path)}')
    conceptnet_graph = gt.load_graph(str(graph_path))
    logger.info(f'Loaded conceptnet graph - {str(graph_path)}')
    remove_self_loops(conceptnet_graph)
    conceptnet_graph.reindex_edges()
    logger.info(f'Generate aspect name to vertex mapping  - {str(graph_path)}')
    vertices_conceptnet = dict(zip(conceptnet_graph.vertex_properties['aspect_name'], conceptnet_graph.vertices()))
    return Graph(conceptnet_graph), vertices_conceptnet
