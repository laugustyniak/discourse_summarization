from pathlib import Path
from typing import Union, Set

import graph_tool as gt
import pandas as pd
from tqdm import tqdm

from aspects.utilities.settings import CONCEPTNET_CSV_EN_PATH, CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_EN_PATH

NUCLEUS_SATELLITE_RELATIONS = {
    'ReceivesAction',
    'HasA',
    'UsedFor',
    'CapableOf',
    'MadeOf',
    # 'HasSubevent',
    # 'CreatedBy',
    # 'DefinedAs',
}
SATELLITE_NUCLEUS_RELATIONS = {
    'PartOf',
    'IsA',
    # 'MannerOf',
    # 'AtLocation',
    # 'Causes',
    # 'DerivedFrom',
}


def generate_english_graph(graph_path: Union[str, Path], relation_types: Set[str] = None) -> gt.Graph:
    df = pd.read_csv(CONCEPTNET_CSV_EN_PATH, index_col=0)

    if relation_types is not None:
        df = df[df.relation.isin(relation_types)]

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
        if row.relation in NUCLEUS_SATELLITE_RELATIONS:
            e = g.add_edge(vertices[row.source], vertices[row.target])
        else:
            e = g.add_edge(vertices[row.target], vertices[row.source])
        e_relation[e] = row.relation

    g.edge_properties['relation'] = e_relation

    # serialize
    g.save(str(graph_path))

    return g


def load_english_hierarchical_graph() -> gt.Graph:
    if CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_EN_PATH.exists():
        return gt.load_graph(CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_EN_PATH.as_posix())
    else:
        return generate_english_graph(
            graph_path=CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_EN_PATH,
            relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS)
        )


if __name__ == '__main__':
    load_english_hierarchical_graph()
