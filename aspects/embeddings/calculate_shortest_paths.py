import networkx as nx
from networkx.algorithms import shortest_paths as nx_sp


def calculate_shortest_paths_lengths(graph):
    if nx.is_directed(graph):
        graph = nx.DiGraph(graph)
    else:
        graph = nx.Graph(graph)

    return dict(nx_sp.shortest_path_length(graph))
