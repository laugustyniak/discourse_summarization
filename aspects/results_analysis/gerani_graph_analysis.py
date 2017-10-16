import logging
import numpy as np

log = logging.getLogger(__name__)


def get_dir_moi_for_node(graph, aspects_per_edu, documents_info):
    aspects_per_edu_sentiment = []
    for doc_info in documents_info.itervalues():
        for edu_id, sentiment in doc_info['sentiment'].iteritems():
            try:
                for aspect in aspects_per_edu[edu_id]:
                    aspects_per_edu_sentiment.append((aspect, sentiment))
            except KeyError as err:
                log.info('There is not aspect: {} in graph'.format(str(err)))

    aspect_sentiments = dict()
    for k, v in aspects_per_edu_sentiment:
        aspect_sentiments.setdefault(k, list()).append(v)

    for aspect, sentiments in aspect_sentiments.iteritems():
        try:
            graph.node[aspect]['sentiment'] = sentiments
            graph.node[aspect]['dir_moi'] = np.sum([x ** 2 for x in sentiments])
        except KeyError as err:
            log.info('There is not aspect: {} in graph'.format(str(err)))

    return graph
