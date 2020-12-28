import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from typing import Dict, Sequence, Callable, List

import pandas as pd
import spacy
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


nlp = spacy.load("en_vectors_web_lg")


def cluster_embeddings_with_spacy(df: pd.DataFrame, n_clusters: int) -> Dict[str, str]:
    df["embedding"] = [
        nlp(text).vector for text in tqdm(df.text.tolist(), desc="Generate embeddings")
    ]

    df["cluster"] = (
        AgglomerativeClustering(n_clusters=n_clusters)
        .fit(df.embedding.tolist())
        .labels_
    )

    aspect_cluster_representants = {}
    for group, sub_df in df.groupby(by=["cluster"]):
        representant = sub_df.sort_values(by="importance").head(1)["text"].iloc[0]
        for aspect in sub_df.text.tolist():
            aspect_cluster_representants[aspect] = representant

    return aspect_cluster_representants


def parallelized_extraction(
    elements: Sequence, fn: Callable, batch_size: int, desc: str = "Running in parallel"
) -> List:
    with ProcessPoolExecutor(multiprocessing.cpu_count()) as pool:
        return list(
            tqdm(
                pool.map(fn, elements, chunksize=batch_size),
                total=len(elements),
                desc=desc,
            )
        )
