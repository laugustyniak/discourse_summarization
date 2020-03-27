from functools import lru_cache

import graph_tool as gt
import pandas as pd

from aspects.utilities.settings import CONCEPTNET_CSV_EN_PATH, CONCEPTNET_GRAPH_TOOL_EN_PATH


def generate_english_graph() -> gt.Graph:
    df = pd.read_csv(CONCEPTNET_CSV_EN_PATH, index_col=0)

    graph = gt.Graph()
    relation = graph.new_edge_property('string')
    # df.values creates numpy array of edgelist
    graph.add_edge_list(df.values, hashed=True, eprops=[relation])

    # serialize
    graph.save(CONCEPTNET_GRAPH_TOOL_EN_PATH.as_posix())

    return graph


@lru_cache(1)
def load_english_graph() -> gt.Graph:
    if CONCEPTNET_GRAPH_TOOL_EN_PATH.exists():
        return gt.load_graph(CONCEPTNET_GRAPH_TOOL_EN_PATH.as_posix())
    else:
        return generate_english_graph()
