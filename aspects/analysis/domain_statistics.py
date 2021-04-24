from collections import Counter
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from more_itertools import flatten
from tqdm import tqdm


def generate_domain_statistics(
    reviews_path: Path, min_trees: int = 200
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    statistics = []
    dfs = {}

    for discourse_trees_df_path in tqdm(
        list(reviews_path.glob("review*/discourse_trees_df.pkl"))
    ):
        try:
            df = pd.read_pickle(discourse_trees_df_path)

            if "aspects" in df.columns and len(df) > min_trees:
                dataset_name = (
                    discourse_trees_df_path.parent.stem.replace("reviews", "")
                    .replace("_", " ")
                    .replace("-", " ")
                    .replace("docs", "")
                    .strip()
                )

                dfs[dataset_name] = df

                aspects_in_tuples = get_tuples_aspects(df)

                all_aspects_df = pd.DataFrame(
                    Counter(
                        flatten(df.aspects.apply(flatten).apply(list).tolist())
                    ).most_common(),
                    columns=["aspect", "aspect_occurrences"],
                )

                statistics.append(
                    {
                        "Dataset Name": dataset_name,
                        "# of reviews": df.shape[0],
                        "Average length of reviews (# of words)": df.text.apply(
                            lambda review: len(review.split())
                        ).mean(),
                        "Average number of EDUs": df.edus.apply(
                            lambda edus: len(edus)
                        ).mean(),
                        "# of unique aspects": len(
                            set(flatten(df.aspects.apply(flatten).apply(list).tolist()))
                        ),
                        "# of all aspects": len(
                            list(
                                flatten(df.aspects.apply(flatten).apply(list).tolist())
                            )
                        ),
                        "# of unique aspects in tuples": aspects_in_tuples.shape[0],
                        "# of aspects in tuples (aspect frequency >= 10)": aspects_in_tuples[
                            aspects_in_tuples.aspect_occurrences >= 10
                        ].shape[
                            0
                        ],
                        "# of all aspects (aspect frequency >= 10)": all_aspects_df[
                            all_aspects_df.aspect_occurrences >= 10
                        ].shape[0],
                        "# of DTs with at least one pair of aspects": df[
                            df.aspects.apply(
                                lambda aspects: list(flatten(aspects))
                            ).apply(lambda aspects: len(set(aspects)) > 1)
                        ]["aspects"].shape[0],
                        "# of DTs with a single aspect": df[
                            df.aspects.apply(
                                lambda aspects: list(flatten(aspects))
                            ).apply(lambda aspects: len(set(aspects)) == 1)
                        ]["aspects"].shape[0],
                        "Average number of aspect tuples from each DT": df.rules.apply(
                            lambda rules: len(rules)
                        ).mean(),
                        "Minimum depth of DT": df.discourse_tree.apply(
                            lambda dt: dt.height()
                        ).min(),
                        "Average depth of DT": df.discourse_tree.apply(
                            lambda dt: dt.height()
                        ).mean(),
                        "Maximum depth of DT": df.discourse_tree.apply(
                            lambda dt: dt.height()
                        ).max(),
                    }
                )
            else:
                print(f"Not fully processed dataset: {discourse_trees_df_path}")

        except Exception as e:
            print(f"Problem with parsing {discourse_trees_df_path} - {str(e)}")

    return pd.DataFrame(statistics), dfs


def get_tuples_aspects(df):
    aspects_in_tuples_counter = Counter(
        flatten(
            flatten(
                (
                    df[
                        df.aspects.apply(lambda aspects: list(flatten(aspects))).apply(
                            lambda aspects: len(set(aspects)) > 1
                        )
                    ]["aspects"]
                )
            )
        )
    )
    aspects_in_tuples_counter_df = pd.DataFrame(
        aspects_in_tuples_counter.most_common(),
        columns=["aspect", "aspect_occurrences"],
    )
    return aspects_in_tuples_counter_df
