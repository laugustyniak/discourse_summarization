import networkx as nx
from NetworkxD3 import simpleNetworkx

G = nx.Graph()
H = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
G.add_nodes_from(H)
G.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("A", "J"), ("B", "E"), ("B", "F"),
                  ("C", "G"), ("C", "H"), ("D", "I")])

simpleNetworkx(G)
