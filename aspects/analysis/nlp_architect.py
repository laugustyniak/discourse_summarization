import logging
import pickle
from collections import namedtuple

import pandas as pd
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

logger = logging.getLogger(__name__)

Metrics = namedtuple('Metrics', 'precision, recall, f1')

MODELS_TO_SKIP = ['char-bilstm', 'char-lstm']

REINDEX_RESULTS_ORDER = (
    'Wo-LSTM',
    'WoCh-LSTM',
    'Wo-LSTM-CRF',
    'WoCh-LSTM-CRF',
    'Wo-BiLSTM',
    'WoCh-BiLSTM',
    'Wo-BiLSTM-CRF',
    'WoCh-BiLSTM-CRF'
)

EMBEDDING_NAMES = {
    'wiki-news-300d-1M': 'fatText-2',
    'glove.42B.300d': 'Glove 42B',
    'crawl-300d-2M': 'fastText',
    'sentic2vec': 'Amazon Reviews',
    'glove.6B.100d': 'Glove 6B 100',
    'numberbatch-en': 'numberbatch',
    'glove.6B.50d': 'Glove 6B 50',
    'glove.840B.300d': 'Glove 840B',
    'glove.6B.200d': 'Glove 6B 200',
    'glove.6B.300d': 'Glove 6B 300',
    'GoogleNews-vectors-negative300': 'word2vec'
}

METHOD_NAMES = {
    'word lstm': 'Wo-LSTM',
    'word lstm crf': 'Wo-LSTM-CRF',
    'char word lstm': 'WoCh-LSTM',
    'char word lstm crf': 'WoCh-LSTM-CRF',
    'word bilstm': 'Wo-BiLSTM',
    'word bilstm crf': 'Wo-BiLSTM-CRF',
    'char word bilstm': 'WoCh-BiLSTM',
    'char word bilstm crf': 'WoCh-BiLSTM-CRF',
}


def _get_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).stem


def _get_metrics(metrics_eval):
    return Metrics(*metrics_eval[0])


def get_metrics(models_paths: Iterable[Path], filter_datasets: str):
    models_metrics = {}
    models_paths = [mp for mp in models_paths if filter_datasets in mp.name.lower()]
    logger.info(f'#{len(models_paths)} models paths have been found')
    for model_path in models_paths:
        model_info = pickle.load(model_path.open('rb'))
        models_metrics[_get_dataset_name(model_path.as_posix())] = _get_metrics(model_info['eval'])
    return models_metrics


def get_models_attributes(models_paths: Iterable[Path], *args):
    models_metrics = {}
    for model_path in models_paths:
        model_info = pickle.load(model_path.open('rb'))
        models_metrics[_get_dataset_name(model_path.as_posix())] = {
            k: model_info[k]
            for k
            in args
        }
    return models_metrics


def get_models_params_from_name(model_name: str):
    """
    Parse names such as: model-info-char-bilstm-crf-10epochs-laptops-train.conll to model features and
    neural network architecture only
    """
    return ' '.join(model_name.split('-')[2:-3])


def merge_embeddings_and_architectures_results(result_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    results = {
        embedding: {
            architecture: [metric]
            for architecture, metric
            in architecture_metric.items()
        }
        for embedding, architecture_metric
        in result_dfs[0].transpose().to_dict(orient='index').items()
    }

    for result_df in result_dfs[1:]:
        for embedding, architecture_metric in result_df.transpose().to_dict(orient='index').items():
            if embedding in result_df:
                for architecture, metric in [(a, m) for a, m in architecture_metric.items() if str(m).lower() != 'nan']:
                    results[embedding][architecture] += [metric]
            else:
                results[embedding] = architecture_metric

    return pd.DataFrame(results)


def skip_char_models(model_name: Path):
    return not any(m in model_name.as_posix() for m in MODELS_TO_SKIP)


def get_models_metric(
        all_models_path: Path,
        filter_datasets: str,
        embedding_names: Dict,
        metric_name: str = 'f1',
        reindex_results_order: Tuple[str] = None
):
    if reindex_results_order is None:
        reindex_results_order = REINDEX_RESULTS_ORDER
    model_metric_by_word_embedding = {}

    for word_embedding_models_path in all_models_path.glob('*'):
        models_metric = {}
        models_paths = list(filter(skip_char_models, word_embedding_models_path.glob('*')))
        models_paths = [m for m in models_paths if 'tensorboard' not in m.as_posix()]
        models_metrics = get_metrics(models_paths, filter_datasets)

        for model_name, model_metrics in models_metrics.items():
            model_name = get_models_params_from_name(model_name)
            models_metric[model_name] = model_metrics._asdict()[metric_name]

        if models_metric:
            model_metric_by_word_embedding[embedding_names[word_embedding_models_path.name]] = models_metric

    df = pd.DataFrame.from_dict(model_metric_by_word_embedding).round(2)
    df = df.transpose().rename(index=str, columns=METHOD_NAMES).transpose()
    return df.reindex(reindex_results_order)


if __name__ == '__main__':
    # atts = get_models_attributes(
    #     list(
    #         Path('/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models/glove.840B.300d/').glob(
    #             '*')),
    #     'predictions', 'y_test')
    df_oxygen_1 = get_models_metric(
        all_models_path=Path(
            '/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models-oxygen-1/models/'),
        filter_datasets='laptops',
        embedding_names=EMBEDDING_NAMES,
        metric_name='precision'
    )
    pass
