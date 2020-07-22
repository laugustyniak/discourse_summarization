from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Union, Set, List

import graph_tool as gt
import pandas as pd
from graph_tool.stats import remove_self_loops
from tqdm import tqdm

from aspects.utilities import settings

NUCLEUS_SATELLITE_RELATIONS = {
    'ReceivesAction',
    'HasA',
    'UsedFor',
    'CapableOf',
    'MadeOf',
    # 'HasSubevent',
    'CreatedBy',
    'DefinedAs',
}
SATELLITE_NUCLEUS_RELATIONS = {
    'PartOf',
    'IsA',
    'MannerOf',
    'AtLocation',
    'Causes',
    'DerivedFrom',
}


def generate_english_graph(
        graph_path: Union[str, Path],
        relation_types: Set[str] = None,
) -> gt.Graph:
    df = pd.read_csv(settings.CONCEPTNET_CSV_EN_PATH, index_col=0)

    synonyms_df = df[df.relation == 'Synonym']
    all_synonyms = list(set(synonyms_df.target.tolist() + synonyms_df.source.tolist()))

    synonyms = defaultdict(list)
    for s, t in tqdm(
            zip(synonyms_df.source.tolist(), synonyms_df.target.tolist()),
            desc='Generating synonyms mapping',
            total=len(synonyms_df)
    ):
        s = str(s)
        t = str(t)
        synonyms[s] += [t]
        synonyms[t] += [s]

    synonyms = {
        k: set(v).union({k})
        for k, v in synonyms.items()
    }

    if relation_types is not None:
        df = df[df.relation.isin(relation_types)]

    # extend relation with synonyms hierarchical types

    g = gt.Graph()
    e_relation = g.new_edge_property('string')
    v_aspect_name = g.new_vertex_property('string')

    all_vertices_names = set(df.source.tolist() + df.target.tolist() + all_synonyms)

    vertices = {}
    for aspect_name in tqdm(all_vertices_names, desc='Vertices adding to the graph...'):
        v = g.add_vertex()
        vertices[aspect_name] = v
        v_aspect_name[v] = aspect_name

    g.vertex_properties['aspect_name'] = v_aspect_name

    def get_synonyms(vertex_name) -> List[str]:
        return synonyms[vertex_name] if vertex_name in synonyms else [vertex_name]

    edge_adding_errors = 0
    for row in tqdm(df.itertuples(), desc='Edges adding to the graph...', total=len(df)):
        source = row.source
        target = row.target

        # TODO: add other possible hierarchical relations from conceptnet
        if row.relation in SATELLITE_NUCLEUS_RELATIONS:
            source, target = target, source

        for s, t in product(get_synonyms(source), get_synonyms(target)):
            try:
                e = g.add_edge(vertices[s], vertices[t])
                e_relation[e] = row.relation
            except KeyError:
                edge_adding_errors += 1

    print(f'{edge_adding_errors} edges with errors skipped')

    g.edge_properties['relation'] = e_relation

    # serialize
    g.save(str(graph_path))

    return g


def prepare_conceptnet_graph(graph_path: str, relation_types: Set[str]):
    g = gt.load_graph(graph_path)
    remove_self_loops(g)
    g.reindex_edges()

    # filter relations
    e_hierarchical_relation_filter = g.new_edge_property('bool')
    relations = list(g.properties[('e', 'relation')])
    for edge, edge_relation in tqdm(
            zip(
                g.edges(),
                relations
            ),
            desc='Edge filtering...',
            total=len(relations)
    ):
        e_hierarchical_relation_filter[edge] = edge_relation in relation_types
    g.set_edge_filter(e_hierarchical_relation_filter)

    vertices = dict(zip(g.vertex_properties['aspect_name'], g.vertices()))

    return g, vertices


if __name__ == '__main__':
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_ALL_RELATIONS_WITH_SYNONYMS_EN_PATH,
        relation_types=None
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_ALL_RELATIONS_WITHOUT_SYNONYMS_EN_PATH,
        relation_types=None
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITH_SYNONYMS_EN_PATH,
        relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS)
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITHOUT_SYNONYMS_EN_PATH,
        relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS)
    )