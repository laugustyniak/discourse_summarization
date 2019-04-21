import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utilities import settings


def get_embedding_vocab(word_embedding_path=None, word_embedding_vocab_path=None):
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
        in enumerate(word_embedding)
    }
    with open(word_embedding_vocab_path.as_posix(), 'wb') as f:
        pickle.dump(vocab, f)

    return vocab


def load_word_embeddings(file_path):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
    file_path = str(file_path)
    with open(file_path, encoding='utf-8') as fp:
        word_vectors = {}
        size = None
        try:
            for line in tqdm(fp, desc=file_path + ': embedding loading'):
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
    return word_vectors, size
