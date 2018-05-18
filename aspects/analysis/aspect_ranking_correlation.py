from os.path import basename

from scipy.stats import stats

from aspects.analysis.statistics_bing_liu import get_aspect_frequency_ranking
from aspects.analysis.statistics_rst_graphs import get_aspects_rankings_from_rst, ASPECTS_GRAPH_PATHS, \
    get_aspect_ranking_based_on_rst_and_pagerank
from aspects.utilities import settings


def get_spearman_and_kendalltau_correlations(top_n_aspects: int = 10):
    correlations = {}

    for reviews_path in settings.ALL_BING_LIU_REVIEWS_PATHS:
        dataset_name = basename(reviews_path).split('.')[0]
        print(f'Dataset to analyze: {dataset_name}')

        # get freq aspects from Bing Liu manually created datasets
        aspects_freq_manual_assignment = get_aspect_frequency_ranking(reviews_path=reviews_path, top_n=top_n_aspects)
        print(aspects_freq_manual_assignment)

        # get aspects from RST + PageRank
        aspects_from_rst_based_on_pagerank = get_aspects_rankings_from_rst(
            [
                aspects_graph_path
                for aspects_graph_path
                in ASPECTS_GRAPH_PATHS
                if dataset_name in aspects_graph_path
            ][0],
            aspects_freq_manual_assignment
        )

        aspects_from_rst_based_on_pagerank_top = get_aspect_ranking_based_on_rst_and_pagerank(
            [
                aspects_graph_path
                for aspects_graph_path
                in ASPECTS_GRAPH_PATHS
                if dataset_name in aspects_graph_path
            ][0],
            top_n_aspects
        )

        print(f'Bing Liu aspects: {aspects_from_rst_based_on_pagerank}')
        print(f'RST aspects: {aspects_from_rst_based_on_pagerank_top}')

        aspects_freq_manual_assignment_ranking, aspects_from_rst_based_on_pagerank_ranking = create_rankings(
            aspects_freq_manual_assignment, aspects_from_rst_based_on_pagerank)

        spearman_correlation = stats.spearmanr(
            aspects_freq_manual_assignment_ranking, aspects_from_rst_based_on_pagerank_ranking)
        print(f'{dataset_name}, Spearman correlation of ranking: {spearman_correlation}')

        kendalltau_correlation = stats.kendalltau(
            aspects_freq_manual_assignment_ranking, aspects_from_rst_based_on_pagerank_ranking)
        print(f'{dataset_name}, KendalTau correlation of ranking: {kendalltau_correlation}')

        aspects_manual = set(aspects_freq_manual_assignment)
        aspects_rst = set(aspects_from_rst_based_on_pagerank_top)

        correlations[dataset_name] = {
            'Spearman Correlation': spearman_correlation[0],
            'Spearman p-value': spearman_correlation[1],
            'Kendall Tau Correlation': kendalltau_correlation[0],
            'Kendall Tau p-value': kendalltau_correlation[1],
            'Jaccard': len(aspects_manual.intersection(aspects_rst))/len(aspects_manual.union(aspects_rst)),
            'Precision': len(aspects_manual.intersection(aspects_rst))/len(aspects_manual),
            # 'Recall': len(aspects_manual.intersection(aspects_rst))/len(aspects_manual.union(aspects_rst))
        }
    return correlations


def create_rankings(aspects_1, aspects_2):
    aspects_rank = {aspect: idx for idx, aspect in enumerate(aspects_1, start=1)}
    aspects_1_ranking = list(aspects_rank.values())

    aspects_2_ranking = [
        aspect
        for aspect
        in aspects_2
    ]

    return aspects_1_ranking, aspects_2_ranking


if __name__ == '__main__':
    correlations = get_spearman_and_kendalltau_correlations()
