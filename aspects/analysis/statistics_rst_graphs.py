from collections import namedtuple

import networkx as nx
import pandas as pd

from aspects.utilities import settings

ASPECTS_GRAPH_PATHS = [
    'results/ipod/aspects_graph.gpkl',
    'results/Diaper Champ/aspects_graph.gpkl',
    'results/norton/aspects_graph.gpkl',
    'results/Linksys Router/aspects_graph.gpkl',
    'results/MicroMP3/aspects_graph.gpkl',
    'results/Canon S100/aspects_graph.gpkl',
    'results/Canon PowerShot SD500/aspects_graph.gpkl',
    'results/Nokia 6600/aspects_graph.gpkl',
    'results/Hitachi router/aspects_graph.gpkl',
]

ASPECTS_GRAPH_PATHS = [
    (settings.ROOT_PATH.parent / aspects_graph_path).as_posix()
    for aspects_graph_path
    in ASPECTS_GRAPH_PATHS
]

AspectPageRank = namedtuple('Aspect_PageRank', 'aspect, pagerank')


def get_aspect_ranking_based_on_rst_and_pagerank(aspects_graph_path, top_n=10):
    """ Get Data Frame with aspects and its ranking based on page rank of asepct-aspect graph """
    aspect_graph = nx.read_gpickle(aspects_graph_path)

    df = pd.DataFrame(
        [
            AspectPageRank(aspect=x[0], pagerank=x[1])
            for x
            in nx.pagerank_scipy(aspect_graph, weight='weight').items()
        ]
    )
    df.sort_values('pagerank', ascending=False, inplace=True)
    return list(df.aspect.head(top_n))


def get_aspects_rankings_from_rst(aspects_graph_path, aspects):
    aspect_graph = nx.read_gpickle(aspects_graph_path)
    df = pd.DataFrame(
        [
            AspectPageRank(aspect=x[0], pagerank=x[1])
            for x
            in nx.pagerank_scipy(aspect_graph, weight='weight').items()
        ]
    )
    df.sort_values('pagerank', ascending=False, inplace=True)
    df = df.reset_index().drop(['index'], axis=1)
    ranks = [
        get_index_of_element_in_series(df.aspect, aspect)
        for aspect
        in aspects
    ]
    return ranks


def get_index_of_element_in_series(series: pd.Series, element: str) -> int:
    row = series[series == element]
    if not row.empty:
        return row.index[0] + 1
    else:
        return len(series) + 1


if __name__ == '__main__':
    get_aspect_ranking_based_on_rst_and_pagerank(ASPECTS_GRAPH_PATHS[0])
