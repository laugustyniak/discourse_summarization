from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Union, Set, List, Optional

import graph_tool as gt
import pandas as pd
from graph_tool.stats import remove_self_loops
from tqdm import tqdm

from aspects.utilities import settings

"""
ConceptNet Relation Stats
RelatedTo                    1658056
FormOf                        376467 
DerivedFrom                   312305
HasContext                    213983
Synonym                       149378
IsA                           139281
UsedFor                        39401
EtymologicallyRelatedTo        32186
AtLocation                     27290
HasSubevent                    25238
HasPrerequisite                22710
CapableOf                      22677
Antonym                        18450
Causes                         16568
MotivatedByGoal                 9489
SimilarTo                       8741
HasProperty                     8433
ReceivesAction                  6037
HasA                            5545
CausesDesire                    4688
PartOf                          3590
HasFirstSubevent                3347
DistinctFrom                    3315
Desires                         3170
NotDesires                      2886
HasLastSubevent                 2874
dbpedia_genus                   2643
dbpedia_genre                   2261
DefinedAs                       2173
dbpedia_influencedBy            1080
InstanceOf                      1070
dbpedia_language                 669
dbpedia_occupation               633
dbpedia_field                    580
dbpedia_knownFor                 552
MadeOf                           545
dbpedia_capital                  407
dbpedia_product                  329
NotCapableOf                     329
NotHasProperty                   327
CreatedBy                        261
EtymologicallyDerivedFrom        175
dbpedia_leader                    70
LocatedNear                       49
MannerOf                           6
SymbolOf                           4
"""


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
    'HasContext',
}

SYNONYMOUS_RELATIONS = [
    'Synonym',
    'SimilarTo',
    'FormOf'  # grammatical forms, plurals. i.e., floatages	and floatage - second most common conceptnet relation
    # 'RelatedTo'  # sometime they are not so similar in the understainding os synonimity, but this is the most common relation in Conceptnet
]


def generate_english_graph(
        graph_path: Union[str, Path],
        relation_types: Set[str] = None,
        synonymous_relations: Optional[List[str]] = None
) -> gt.Graph:
    df = pd.read_csv(settings.CONCEPTNET_CSV_EN_PATH, index_col=0)

    if synonymous_relations:
        synonyms_df = df[df.relation.isin(synonymous_relations)]
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

    g = gt.Graph()
    e_relation = g.new_edge_property('string')
    v_aspect_name = g.new_vertex_property('string')

    if synonymous_relations:
        all_vertices_names = set(df.source.tolist() + df.target.tolist() + all_synonyms)
    else:
        all_vertices_names = set(df.source.tolist() + df.target.tolist())

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

        if row.relation in SATELLITE_NUCLEUS_RELATIONS:
            source, target = target, source

        if synonymous_relations:
            for s, t in product(get_synonyms(source), get_synonyms(target)):
                try:
                    e = g.add_edge(vertices[s], vertices[t])
                    e_relation[e] = row.relation
                except KeyError:
                    edge_adding_errors += 1
        else:
            e = g.add_edge(vertices[source], vertices[target])
            e_relation[e] = row.relation

    print(f'{edge_adding_errors} edges with errors skipped')
    g.edge_properties['relation'] = e_relation
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
        relation_types=None,
        synonymous_relations=SYNONYMOUS_RELATIONS
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_ALL_RELATIONS_WITHOUT_SYNONYMS_EN_PATH,
        relation_types=None,
        synonymous_relations=None
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITH_SYNONYMS_EN_PATH,
        relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS),
        synonymous_relations=SYNONYMOUS_RELATIONS
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITH_SYNONYMS_AND_RELATED_TO_EN_PATH,
        relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS),
        synonymous_relations=SYNONYMOUS_RELATIONS + ['RelatedTo']
    )
    generate_english_graph(
        graph_path=settings.CONCEPTNET_GRAPH_TOOL_HIERARCHICAL_WITHOUT_SYNONYMS_EN_PATH,
        relation_types=SATELLITE_NUCLEUS_RELATIONS.union(NUCLEUS_SATELLITE_RELATIONS),
        synonymous_relations=None
    )
