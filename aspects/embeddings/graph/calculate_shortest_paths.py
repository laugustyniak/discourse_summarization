from pathlib import Path
from typing import Union

import networkx as nx
from networkx.algorithms import shortest_paths as nx_sp

from aspects.data_io import serializer


def calculate_shortest_paths_lengths(
        graph: Union[nx.Graph, nx.DiGraph],
        shortest_paths_path: Union[str, Path]
):
    if nx.is_directed(graph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    serializer.save(
        dict(nx_sp.shortest_path_length(graph)),
        shortest_paths_path
    )
