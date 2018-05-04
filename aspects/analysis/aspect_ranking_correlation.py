from os.path import basename

from scipy.stats import stats
from sklearn import preprocessing

from aspects.analysis.statistics_bing_liu import get_aspect_frequency_ranking
from aspects.analysis.statistics_rst_graphs import get_aspect_ranking_based_on_rst_and_pagerank, ASPECTS_GRAPH_PATHS
from aspects.utilities import settings


def get_spearman_and_kendalltau_correlations():
    correlations = {}

    for reviews_path in settings.ALL_BING_LIU_REVIEWS_PATHS:
        label_encoder = preprocessing.LabelEncoder()
        dataset_name = basename(reviews_path).split('.')[0]
        print(f'Dataset to analyze: {dataset_name}')

        # get freq aspects from Bing Liu manually created datasets
        aspects_freq_manual_assignment = get_aspect_frequency_ranking(reviews_path)
        print(aspects_freq_manual_assignment)

        # get aspects from RST + PageRank
        aspects_from_rst_based_on_pagerank = get_aspect_ranking_based_on_rst_and_pagerank(
            [
                aspects_graph_path
                for aspects_graph_path
                in ASPECTS_GRAPH_PATHS
                if dataset_name in aspects_graph_path
            ][0]
        )
        print(aspects_from_rst_based_on_pagerank)

        # merge aspects from both sources
        all_aspects_from_both_datasets_rankings = aspects_freq_manual_assignment + aspects_from_rst_based_on_pagerank

        label_encoder.fit(all_aspects_from_both_datasets_rankings)
        aspects_freq_manual_assignment_vector = label_encoder.transform(aspects_freq_manual_assignment)
        aspects_from_rst_based_on_pagerank_vector = label_encoder.transform(aspects_from_rst_based_on_pagerank)

        spearman_correlation = stats.spearmanr(
            aspects_freq_manual_assignment_vector, aspects_from_rst_based_on_pagerank_vector)
        print(f'{dataset_name}, Spearman correlation of ranking: {spearman_correlation}')

        kendalltau_correlation = stats.kendalltau(
            aspects_freq_manual_assignment_vector, aspects_from_rst_based_on_pagerank_vector)
        print(f'{dataset_name}, KendalTau correlation of ranking: {kendalltau_correlation}')

        correlations[dataset_name] = {
            'Spearman Correlation': spearman_correlation[0],
            'Spearman p-value': spearman_correlation[1],
            'Kendall Tau Correlation': kendalltau_correlation[0],
            'Kendall Tau p-value': kendalltau_correlation[1]
        }
    return correlations


if __name__ == '__main__':
    correlations = get_spearman_and_kendalltau_correlations()