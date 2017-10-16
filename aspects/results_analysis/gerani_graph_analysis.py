import logging
import numpy as np

log = logging.getLogger(__name__)


def get_dir_moi_for_node(graph, aspects_per_edu, documents_info):
    n_skipped_edus = 0
    n_aspects_not_in_graph = 0
    n_aspects_updated = 0
    n_all_documents = len(documents_info)
    aspects_per_edu_sentiment = []
    for doc_info in documents_info.itervalues():
        for edu_id, sentiment in doc_info['sentiment'].iteritems():
            try:
                for aspect in aspects_per_edu[edu_id]:
                    aspects_per_edu_sentiment.append((aspect, sentiment))
            except KeyError as err:
                n_skipped_edus += 1
                log.info(
                    'Aspect: {} not extracted from edu: {}'.format(str(err),
                                                                   edu_id))

    aspect_sentiments = dict()
    for k, v in aspects_per_edu_sentiment:
        aspect_sentiments.setdefault(k, list()).append(v)

    for aspect, sentiments in aspect_sentiments.iteritems():
        try:
            graph.node[aspect]['sentiment'] = sentiments
            graph.node[aspect]['dir_moi'] = np.sum([x ** 2 for x in sentiments])
            n_aspects_updated += 1
        except KeyError as err:
            n_aspects_not_in_graph += 1
            log.info('There is not aspect: {} in graph'.format(str(err)))

    log.info('#{} skipped aspects out of #{} documents'.format(n_skipped_edus,
                                                               n_all_documents))
    log.info('#{} aspects not in graph'.format(n_aspects_not_in_graph))
    log.info('#{} aspects updated in graph'.format(n_aspects_updated))

    return graph
