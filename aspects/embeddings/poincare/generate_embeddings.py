import logging
from itertools import product
from typing import NamedTuple

import pandas as pd
from gensim.models.poincare import PoincareModel
from tqdm import tqdm

from aspects.pipelines.aspect_analysis import AspectAnalysis
from aspects.utilities import settings


class AspectsRelation(NamedTuple):
    aspect_1: str
    aspect_2: str
    relation_type: str
    weight: float


logging.basicConfig(level=logging.INFO)

DATASET_PATH = settings.AMAZON_REVIEWS_CELL_PHONES_AND_ACCESSORIES_DATASET_JSON

aspect_analysis_gerani = AspectAnalysis(
    input_path=DATASET_PATH.as_posix(),
    output_path=settings.DEFAULT_OUTPUT_PATH / DATASET_PATH.stem,
    experiment_name='gerani',
    max_docs=50000
)

discourse_tree_df = pd.read_pickle(aspect_analysis_gerani.paths.discourse_trees_df)

relations = []

for row_id, row in tqdm(discourse_tree_df.iterrows(), total=len(discourse_tree_df),
                        desc='Generating aspect-aspect graph based on rules'):
    for edu_left, edu_right, relation, weight in row.rules:
        for aspect_left, aspect_right in product(row.aspects[edu_left], row.aspects[edu_right]):
            relations.append((aspect_left, aspect_right))

model = PoincareModel(train_data=relations, size=2, burn_in=0)

model.train(epochs=100, print_every=500)

model.save(aspect_analysis_gerani.paths.aspects_poincare_embeddings)
