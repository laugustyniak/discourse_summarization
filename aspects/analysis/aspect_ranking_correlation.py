from os.path import basename

import click
from scipy.stats import stats
from sklearn import preprocessing

from aspects.analysis.statistics_bing_liu import get_aspect_frequency_ranking, reviews_paths
from aspects.analysis.statistics_rst_graphs import get_aspect_ranking_based_on_rst_and_pagerank, ASPECTS_GRAPH_PATHS


@click.command()
def get_spearman_and_kendalltau_correlations():
    correlations = {}

    for reviews_path in reviews_paths:
        label_encoder = preprocessing.LabelEncoder()
        dataset_name = basename(reviews_path).split('.')[0]
        click.echo(f'Dataset to analyze: {dataset_name}')

        # get freq aspects from Bing Liu manually created datasets
        aspects_freq_manual_assignment = get_aspect_frequency_ranking(reviews_path)
        click.echo(aspects_freq_manual_assignment)

        # get aspects from RST + PageRank
        aspects_from_rst_based_on_pagerank = get_aspect_ranking_based_on_rst_and_pagerank(
            [
                aspects_graph_path
                for aspects_graph_path
                in ASPECTS_GRAPH_PATHS
                if dataset_name in aspects_graph_path
            ][0]
        )
        click.echo(aspects_from_rst_based_on_pagerank)

        # merge aspects from both sources
        all_aspects_from_both_datasets_rankings = aspects_freq_manual_assignment + aspects_from_rst_based_on_pagerank

        label_encoder.fit(all_aspects_from_both_datasets_rankings)
        aspects_freq_manual_assignment_vector = label_encoder.transform(aspects_freq_manual_assignment)
        aspects_from_rst_based_on_pagerank_vector = label_encoder.transform(aspects_from_rst_based_on_pagerank)

        spearman_correlation = stats.spearmanr(
            aspects_freq_manual_assignment_vector, aspects_from_rst_based_on_pagerank_vector)
        click.echo(f'{dataset_name}, Spearman correlation of ranking: {spearman_correlation}')

        kendalltau_correlation = stats.kendalltau(
            aspects_freq_manual_assignment_vector, aspects_from_rst_based_on_pagerank_vector)
        click.echo(f'{dataset_name}, KendalTau correlation of ranking: {kendalltau_correlation}')

        correlations[dataset_name] = {
            'spearman': spearman_correlation,
            'kendalltau': kendalltau_correlation
        }
    return correlations


if __name__ == '__main__':
    correlations = get_spearman_and_kendalltau_correlations()
