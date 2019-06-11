from pathlib import Path

import pandas as pd
import seaborn as sns
from cytoolz.itertoolz import pluck
from lxml import etree as ET

from utilities.settings import SEMEVAL_DATASETS_2016

sns.set(style="whitegrid")

semeval_files = Path(SEMEVAL_DATASETS_2016).glob('*.xml*')
dfs = []

for semeval_file in list(semeval_files):
    tree = ET.parse(semeval_file.as_posix())
    root = tree.getroot()

    reviews = root.findall("Review")
    sentences = root.findall("**/sentence")
    # print("# Reviews   : ", len(reviews))
    print("# Sentences : ", len(sentences))

    opinions = root.findall("**/**/Opinion")
    categories = [opinion.attrib["category"] for opinion in opinions]
    entities_and_aspects = [cat.split('#') for cat in categories]
    polarities = [opinion.attrib["polarity"] for opinion in opinions]
    print("# Opinions  : ", len(opinions))
    df = pd.DataFrame({
        "category": categories,
        "entity": list(pluck(0, entities_and_aspects)),
        'aspect': list(pluck(1, entities_and_aspects)),
        "polarity": polarities
    }).sort_values("entity")
    dfs.append(df)

df_all = pd.concat(dfs)
df_all.polarity = df_all.polarity.apply(lambda t: t if t else 'unknown')
df_all.to_csv(SEMEVAL_DATASETS_2016 / 'all-entities-and-aspects.csv')

g = sns.countplot(
    y="category",
    hue="polarity",
    data=df_all,
    palette={
        "neutral": "yellow",
        "negative": "#FF1744",
        "positive": "#00E676",
        "unknown": "grey"
    }
)
