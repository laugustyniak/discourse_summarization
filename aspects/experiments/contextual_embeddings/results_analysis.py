from pathlib import Path

import pandas as pd


def get_metrics(model_paths: str, loss_file_name: str = 'loss.tsv', metric_col_name: str = 'TEST_F-SCORE'):
    metrics = {}
    for model_path in Path(model_paths).glob('*'):
        df = pd.read_csv(model_path / loss_file_name, sep='\t')
        metrics[model_path.stem] = df[metric_col_name].tolist()[-1]
    return metrics


if __name__ == '__main__':
    get_metrics('/home/laugustyniak/github/nlp/flair/experiments/resources/taggers')
