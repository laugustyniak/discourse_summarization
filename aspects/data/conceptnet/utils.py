from functools import lru_cache

import graph_tool as gt
import pandas as pd
from tqdm import tqdm

from aspects.utilities.settings import CONCEPTNET_CSV_EN_PATH, CONCEPTNET_GRAPH_TOOL_EN_PATH


def generate_english_graph() -> gt.Graph:
    df = pd.read_csv(CONCEPTNET_CSV_EN_PATH, index_col=0)

    g = gt.Graph()
    e_relation = g.new_edge_property('string')
    v_aspect_name = g.new_vertex_property('string')

    vertices = {}
    for aspect_name in tqdm(set(df.source.tolist() + df.target.tolist()), desc='Vertices adding to the graph...'):
        v = g.add_vertex()
        vertices[aspect_name] = v
        v_aspect_name[v] = aspect_name

    g.vertex_properties['aspect_name'] = v_aspect_name

    for row in tqdm(df.itertuples(), desc='Edges adding to the graph...', total=len(df)):
        e = g.add_edge(vertices[row.source], vertices[row.target])
        e_relation[e] = row.relation

    g.edge_properties['relation'] = e_relation

    # serialize
    g.save(CONCEPTNET_GRAPH_TOOL_EN_PATH.as_posix())

    return g


@lru_cache(1)
def load_english_graph() -> gt.Graph:
    if CONCEPTNET_GRAPH_TOOL_EN_PATH.exists():
        return gt.load_graph(CONCEPTNET_GRAPH_TOOL_EN_PATH.as_posix())
    else:
        return generate_english_graph()
