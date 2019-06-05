from collections import Counter
from pathlib import Path

import pandas as pd
import seaborn as sns
from lxml import etree as ET

from utilities.settings import SEMEVAL_DATASETS_2016

sns.set(style="whitegrid")

semeval_files = Path(SEMEVAL_DATASETS_2016).glob('*.xml*')

for semeval_file in list(semeval_files)[:1]:
    tree = ET.parse(semeval_file.as_posix())
    root = tree.getroot()

    reviews = root.findall("Review")
    sentences = root.findall("**/sentence")
    # print("# Reviews   : ", len(reviews))
    print("# Sentences : ", len(sentences))

    opinions = root.findall("**/**/Opinion")
    categories = [opinion.attrib["category"] for opinion in opinions]
    polarities = [opinion.attrib["polarity"] for opinion in opinions]
    print("# Opinions  : ", len(opinions))
    df = pd.DataFrame({"categories": categories, "polarities": polarities}).sort_values("categories")
    categories = Counter(df.categories)
    print("# Categories: ", len(categories))

    g = sns.countplot(y="categories", hue="polarities", data=df,
                      palette={"neutral": "yellow", "negative": "#FF1744", "positive": "#00E676"})

    # sns.plt.tight_layout()
    # sns.plt.savefig(join("images", "1.png"))
    # sns.plt.show()
    #
    # template = Template(open(join("report", "README.md.tmp")).read())
    # data = {}
    # data["num_reviews"] = len(reviews)
    # data["num_sentences"] = len(sentences)
    # data["num_opinions"] = len(opinions)
    # data["num_categories"] = len(categories)
    # content = template.render(data=data)
    # open("README.md", "w").write(content)
