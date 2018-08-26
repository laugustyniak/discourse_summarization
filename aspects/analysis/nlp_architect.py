import pickle
from collections import namedtuple
from pathlib import Path
from typing import Callable, Iterable

Metrics = namedtuple('Metrics', 'precision, recall, f1')


def _get_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).stem


def _get_metrics(metrics_eval):
    return Metrics(*metrics_eval[0])


def filter_datasets(dataset_path: Path, subphrase: str = 'restaurants'):
    return True if subphrase in dataset_path.as_posix() else False


def get_metrics(models_paths: Iterable[Path], filter_datasets: Callable = filter_datasets):
    models_metrics = {}
    for model_path in list(filter(filter_datasets, models_paths)):
        model_info = pickle.load(model_path.open('rb'))
        models_metrics[_get_dataset_name(model_path.as_posix())] = _get_metrics(model_info['eval'])
    return models_metrics


if __name__ == '__main__':
    metrics = get_metrics(Path(
        '/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/models-glove.840B.300d/').glob('*'))
    pass
