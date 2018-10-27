import logging
import operator
from collections import namedtuple

logger = logging.getLogger(__name__)

AspectHierarchyRST = namedtuple('AspectHierarchyRST', 'nucleus, satellite, all_relations')


def get_aspects_hierarchy(edges_counter, aspects_page_ranks, aspect_1, aspect_2):
    """

    :param edges_counter: Counter of edges pairs
    :param aspects_page_ranks: dcit of aspect and it's pagerank value
    :param aspect_1: str
    :param aspect_2: str
    :return:
    """
    logger.info('{} -> {}'.format(aspect_1, aspect_2), edges_counter[(aspect_1, aspect_2)])
    logger.info('{} -> {}'.format(aspect_2, aspect_1), edges_counter[(aspect_2, aspect_1)])

    logger.info(
        'Page Ranks: ',
        aspect_1, aspects_page_ranks[aspect_1],
        aspect_2, aspects_page_ranks[aspect_2]
    )

    pairs = sorted({
        (aspect_1, aspect_2): edges_counter[(aspect_1, aspect_2)],
        (aspect_2, aspect_1): edges_counter[(aspect_2, aspect_1)],
    }, key=operator.itemgetter(1), reverse=True)

    potential_nucleus = pairs[0][1]
    potential_satellite = pairs[0][0]

    if aspects_page_ranks[potential_nucleus] > aspects_page_ranks[potential_satellite]:
        logger.info('S: ', potential_satellite, 'N: ', potential_nucleus)
    else:
        logger.info(
            'S: ', potential_satellite,
            'N: ', potential_nucleus,
            'reversing because Page Rank of potential satellite it greater than potential nucleus')
        # reverse nucleus and setallite
        potential_nucleus = pairs[0][0]
        potential_satellite = pairs[0][1]

    return AspectHierarchyRST(
        potential_nucleus,
        potential_satellite,
        edges_counter[(aspect_1, aspect_2)] + edges_counter[(aspect_2, aspect_1)]
    )
