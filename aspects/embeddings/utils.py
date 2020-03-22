import pickle
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
from gensim.models.keyedvectors import Vocab
from gensim.models.utils_any2vec import _save_word2vec_format
from tqdm import tqdm

from aspects.utilities import settings


def get_word_embedding_vocab(word_embedding_path=None, word_embedding_vocab_path=None):
    if word_embedding_path is None:
        word_embedding_path = settings.WORD_EMBEDDING_GLOVE_42B
    if word_embedding_vocab_path is None:
        word_embedding_vocab_path = settings.WORD_EMBEDDING_GLOVE_42B_VOCAB

    word_embedding_vocab_path = Path(word_embedding_vocab_path)

    if word_embedding_vocab_path.exists():
        with open(word_embedding_vocab_path.as_posix(), 'rb') as f:
            return pickle.load(f)

    word_embedding, word_embedding_vector_size = load_word_embeddings(word_embedding_path)
    vocab = {
        word.lower(): i
        for i, word
        in enumerate(word_embedding, start=2)
    }
    with open(word_embedding_vocab_path.as_posix(), 'wb') as f:
        pickle.dump(vocab, f)

    return vocab


@lru_cache(1)
def load_word_embeddings(file_path):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
    file_path = Path(file_path)

    if file_path.with_suffix('.cache').exists():
        with open(file_path.with_suffix('.cache').as_posix(), 'rb') as f:
            word_vectors = pickle.load(f)
        return word_vectors, len(next(iter(word_vectors.values())))

    with open(file_path.as_posix(), encoding='utf-8') as fp:
        word_vectors = {}
        size = None
        try:
            for line in tqdm(fp, desc=file_path.as_posix() + ': embedding loading'):
                line_fields = line.split()
                if len(line_fields) < 5:
                    continue
                else:
                    if line[0] == ' ':
                        word_vectors[' '] = np.asarray(line_fields, dtype='float32')
                    else:
                        word = str(line_fields[0])
                        try:
                            word_embedding = [float(embedding) for embedding in line_fields[1:]]
                        except:
                            continue
                        word_vectors[word] = np.asarray(word_embedding, dtype='float32')
                        if size is None:
                            size = len(line_fields[1:])
        except UnicodeDecodeError:
            pass

    with open(file_path.with_suffix('.cache').as_posix(), 'wb') as f:
        pickle.dump(file_path.with_suffix('.cache').as_posix(), f)

    return word_vectors, size


def convert_graph_embedding_to_gensim(
        model_path: Union[str, Path],
        dataset_path: Union[str, Path],
        embedding_path: Union[str, Path] = None
) -> Path:
    """
    Example usage
    convert_graph_embedding_to_gensim(
        model_path=(
                DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs' / 'our' / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.model').as_posix(),
        dataset_path=(
                DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs' / 'our' / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.dataset').as_posix()
    )
    """
    if embedding_path is None:
        embedding_path = Path(model_path).with_suffix('.word2vec_format.bin')

    model = torch.load(model_path)
    dataset = torch.load(dataset_path)

    _save_word2vec_format(
        embedding_path,
        vocab={
            word.replace(' ', '_'): Vocab(index=index, count=1)
            for word, index
            in dataset[0].nodes_mapping[0].items()
        },
        vectors=model.embedding.weight.detach().numpy(),
        binary=True
    )

    return embedding_path
