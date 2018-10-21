import pickle
from collections import namedtuple

from pathlib import Path
from typing import Callable, Iterable

Metrics = namedtuple('Metrics', 'precision, recall, f1')


def _get_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).stem


def _get_metrics(metrics_eval):
    return Metrics(*metrics_eval[0])


def filter_datasets(dataset_path: Path, subphrase: str = 'Laptops'):
    return True if subphrase in dataset_path.as_posix() else False


def get_metrics(models_paths: Iterable[Path], filter_datasets: Callable = filter_datasets):
    models_metrics = {}
    for model_path in list(filter(filter_datasets, models_paths)):
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


if __name__ == '__main__':
    atts = get_models_attributes(
        list(
            Path('/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models/glove.840B.300d/').glob(
                '*')),
        'predictions', 'y_test')
    pass
