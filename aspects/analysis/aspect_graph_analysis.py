#!/usr/bin/env python3
import json
from collections import Counter
from datetime import datetime
from operator import itemgetter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.readwrite import json_graph
from pathlib import Path
from typing import Union


def draw_degree_distribution(aspect_graph_path: Union[str, Path], degree_threshold: int = 5):
    start_time = datetime.now()

    dataset_name = Path(aspect_graph_path).stem

    if 'gpkl' in aspect_graph_path.as_posix():
        aspect_graph = nx.read_gpickle(aspect_graph_path)
    elif 'gexf' in aspect_graph_path.as_posix():
        aspect_graph = nx.read_gexf(aspect_graph_path)
    else:
        raise Exception('Wrong graph type')

    degree_sequence = sorted([d for n, d in aspect_graph.degree()], reverse=True)  # degree sequence
    degree_count = {
        degree: count
        for degree, count
        in Counter(degree_sequence).items()
        if count > degree_threshold
    }
    deg, cnt = zip(*degree_count.items())
    fig, ax = plt.subplots(figsize=(30, 10))
    plt.bar(deg, cnt, width=0.845, color='b')
    plt.title(f'Degree Histogram')
    plt.ylabel('Count')
    plt.xlabel('Degree')
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

    # PageRank
    page_ranks = sorted(nx.pagerank_scipy(aspect_graph, weight='weight').items(), key=itemgetter(1), reverse=True)
    pd.Series([x[1] for x in page_ranks]).plot(kind='hist')
    plt.title(f'PageRank Histogram {dataset_name}')
    plt.ylabel('Count')
    plt.xlabel('PageRank')

    # draw network with PageRank as node size
    print(f'Start PR drawing {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    nodes = []
    values = []
    for node, val in nx.pagerank_scipy(aspect_graph).items():
        nodes.append(node)
        values.append(val * 1000000)
    plt.figure(figsize=(60, 30))
    pos = nx.spring_layout(aspect_graph, k=5)
    nx.draw(aspect_graph,
            pos=pos,
            with_labels=True,
            font_size=24,
            font_weight='bold',
            nodelist=nodes,
            node_size=values,
            arrows=True
            )
    plt.savefig(f'vis/{dataset_name}.png', format="PNG")
    plt.show()

    d = json_graph.node_link_data(aspect_graph)
    with open(f'vis/{dataset_name}.json', 'w') as json_to_dump:
        json.dump(d, json_to_dump)

    print(f'{dataset_name} took {datetime.now() - start_time}s')


if __name__ == '__main__':
    reviews_results_path = Path('../../results/reviews_Cell_Phones_and_Accessories/')
    draw_degree_distribution(reviews_results_path / 'aspects_graph.gpkl')
