import torch

from aspects.data_io import serializer
from aspects.graph.networkx.calculate_shortest_paths import calculate_shortest_paths_lengths
from aspects.embeddings.graph.utils import preprocess_data, calculate_reconstruction_metrics
from aspects.utilities.settings import DEFAULT_OUTPUT_PATH

MAX_NUMBER_OF_NODES = 500


if __name__ == '__main__':
    data_path = DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs' / 'our'

    dataset = torch.load(
        (data_path / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.dataset').as_posix()
    )

    graph_path = data_path / 'aspect_2_aspect_graph.pkl'
    graph = serializer.load(graph_path)

    # sorted_nodes = sorted(list(graph.degree()), key=lambda node_degree_pair: node_degree_pair[1], reverse=True)
    # top_nodes = list(pluck(0, sorted_nodes[:MAX_NUMBER_OF_NODES]))
    # graph = graph.subgraph(top_nodes)

    shortest_paths_path = graph_path.with_suffix('.shortest_paths.pkl')
    if not shortest_paths_path.exists():
        calculate_shortest_paths_lengths(graph, shortest_paths_path)
    graph, graph_dists, mapping = preprocess_data(graph, shortest_paths_path)

    model = torch.load(
        (data_path / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.model').as_posix()
    )
    print(calculate_reconstruction_metrics(
        graph, graph_dists, model.embedding.weight.detach().numpy())
    )
