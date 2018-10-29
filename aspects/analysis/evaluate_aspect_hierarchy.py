from collections import namedtuple

import pandas as pd

from analysis.unsupervised_aspect_hierarchies import amazon_cellphone_aspect_hierarchy_100000_reviews
from enrichments.conceptnets import load_conceptnet_io, get_concept_neighbours_by_relation_type

ParentChildConceptNet = namedtuple('ParentChildConceptNet', 'parent, child, in_conceptnet')


def evalaute_aspect_hierarchy_with_conceptnet(top_n_aspect_pairs=40, jaccard_at=1):
    all_conceptnet_relations = [
        'RelatedTo',

        # 'FormOf',
        # 'IsA',
        # 'PartOf',
        # 'HasA',
        # 'UsedFor',
        # 'AtLocation',
        # 'Synonym',
        # 'SimilarTo',
        # 'MadeOf',

        # 'DefinedAs',
        # 'Entails',
        # 'MannerOf',
        # 'LocatedNear',
        # 'HasContext',
        # 'dbpedia/...',
        # 'ReceivesAction',
        # 'InstanceOf',

        # 'EtymologicallyRelatedTo',
        # 'ExternalURL',
        # 'CapableOf',

        # 'Antonym',
        # 'DistinctFrom',
        # 'DerivedFrom',
        # 'SymbolOf',
        # 'EtymologicallyDerivedFrom',
        # 'CausesDesire',
        # 'Causes',
        # 'HasSubevent',
        # 'HasFirstSubevent',
        # 'HasLastSubevent',
        # 'HasPrerequisite',
        # 'HasProperty',
        # 'MotivatedByGoal',
        # 'ObstructedBy',
        # 'Desires',
        # 'CreatedBy',
    ]

    hierarchical_relations_get_child = [
        'LocatedNear', 'HasA', 'MadeOf',
    ]
    hierarchical_relations_get_parent = [
        'LocatedNear', 'PartOf', 'IsA',
    ]
    synonymity_relations = [
        'Synonym', 'RelatedTo',
    ]
    conceptnet = load_conceptnet_io()

    aspects_checked_in_conceptnet = []
    both_aspects_not_in_conceptnet = []
    for parent_aspect, child_aspect, freq in amazon_cellphone_aspect_hierarchy_100000_reviews[:top_n_aspect_pairs]:
        parent_aspect = parent_aspect.replace(' ', '_')
        child_aspect = child_aspect.replace(' ', '_')
        if parent_aspect in conceptnet and child_aspect in conceptnet:
            neighbours_of_concept_childrens = get_concept_neighbours_by_relation_type(
                conceptnet,
                parent_aspect,
                hierarchical_relations_get_child,
                hierarchical_relations_get_parent,
                synonymity_relations,
                level=jaccard_at
            )
            aspects_checked_in_conceptnet.append(
                ParentChildConceptNet(parent_aspect, child_aspect, child_aspect in neighbours_of_concept_childrens))
        else:
            both_aspects_not_in_conceptnet.append(
                (parent_aspect, parent_aspect in conceptnet, child_aspect, child_aspect in conceptnet))

    n_pairs_in_conceptnet = len(aspects_checked_in_conceptnet) - len(both_aspects_not_in_conceptnet)
    n_correct_hierarchy_pairs = len([x for x in aspects_checked_in_conceptnet if x.in_conceptnet])
    # print('Not in ConceptNet', [x for x in aspects_checked_in_conceptnet if not x.in_conceptnet])
    # print('aspects_checked_in_conceptnet', len(aspects_checked_in_conceptnet))
    jaccard_distance = n_correct_hierarchy_pairs / n_pairs_in_conceptnet
    print('Jaccard@at', jaccard_at, '=', jaccard_distance)
    print('both_aspects_not_in_conceptnet', len(both_aspects_not_in_conceptnet))
    return jaccard_distance


if __name__ == '__main__':
    results = {}
    for n_top_pairs in [5, 10, 20, 30, 40, 50]:
        print('Top #{} pairs'.format(n_top_pairs))
        jaccards = {
            'jaccard@{}'.format(jaccard_at_n): evalaute_aspect_hierarchy_with_conceptnet(n_top_pairs, jaccard_at_n)
            for jaccard_at_n
            in [2, 3]
        }
        results['Top{}'.format(n_top_pairs)] = jaccards

    results_df = pd.DataFrame.from_dict(results).round(2).transpose()
    print(results_df.to_latex())
