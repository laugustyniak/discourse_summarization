import json

import pandas as pd
from tqdm import tqdm

from aspects.aspects.aspects_graph_builder import Aspect2AspectGraph
from aspects.data_io import serializer
from aspects.utilities import settings

REVIEWS_RESULTS = settings.DEFAULT_OUTPUT_PATH.parent.parent / 'results' / 'reviews_Cell_Phones_and_Accessories'

# aspects_per_edu = serializer.load((REVIEWS_RESULTS / 'aspects_per_edu').as_posix())
# edus = serializer.load('results/reviews_Cell_Phones_and_Accessories/raw_edu_list')
# documents_info = serializer.load((REVIEWS_RESULTS / 'documents_info').as_posix())

with open((REVIEWS_RESULTS / 'aspects_per_edu.json').as_posix(), 'r') as f:
    aspects_per_edu = json.load(f)

aspect_relations = serializer.load((REVIEWS_RESULTS / 'edu_dependency_rules').as_posix())

aspect_graph_builder = Aspect2AspectGraph(aspects_per_edu=aspects_per_edu)

aspect_rules = []

for relation in tqdm(aspect_relations.values()):
    for edu_1, edu_2, _, weight in relation:
        for aspect_left, aspect_right in aspect_graph_builder.aspects_iterator(edu_1, edu_2):
            if aspect_left != aspect_right:
                aspect_rules.append((aspect_left, aspect_right, weight))

df = pd.DataFrame(aspect_rules, columns=['id1', 'id2', 'weight'])
df.to_csv(REVIEWS_RESULTS / 'aspect-rules.csv', index=False)

pass
