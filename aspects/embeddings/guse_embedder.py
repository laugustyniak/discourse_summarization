import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tensorflow_hub as hub
from more_itertools import flatten
from more_itertools.more import chunked
from tqdm import tqdm

from aspects.utilities import settings

logger = logging.getLogger(__name__)


def process_to_ids_in_sparse_format(
    sentence_piece_processor: spm.SentencePieceProcessor, sentences: Iterable[str]
) -> Tuple[List, List, Tuple[int, int]]:
    """
    An utility method that processes sentences with the sentence piece processor
    `sp` and returns the results in tf.SparseTensor-similar format:
    (values, indices, dense_shape)
    """

    ids = [sentence_piece_processor.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)

    dense_shape = (len(ids), max_len)
    values = list(flatten(ids))
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]

    return values, indices, dense_shape


class GuseEmbedder:
    """
    Embeds input sentences using Google's Universal Sentence Embedding model

    Attributes:
        embed: function placeholder for GUSE embedding
        sentence_piece_processor: sentence piece processor for changing strings into sparse representation
    """

    def __init__(self, embedding_model_path: str = None):
        """
        Initialize the embed

        :param embedding_model_path: URL of the TensorflowHub model
        """
        if embedding_model_path is None:
            embedding_model_path = settings.ML_GUSE_MODEL_PATH

        self.embed = hub.Module(embedding_model_path)
        self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.computation = self.embed(
            inputs=dict(
                values=self.input_placeholder.values,
                indices=self.input_placeholder.indices,
                dense_shape=self.input_placeholder.dense_shape,
            )
        )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        # this signatures returns the path to the SentencePiece model [tokenizer] required when processing the sentences
        # we must load this model in order to initialize the processor
        sentence_piece_processor_path = self.session.run(
            self.embed(signature="spm_path")
        )

        self.sentence_piece_processor = spm.SentencePieceProcessor()
        self.sentence_piece_processor.Load(sentence_piece_processor_path)

    def embed_sentences(
        self,
        sentences: List[str],
        max_chunk_size: int = 10000,
        partial_save_path: str = None,
    ) -> np.ndarray:
        """
        Embeds a list of sentences

        :param sentences: list of sentences to be embedded
        :param max_chunk_size: size of max size of chunk of sentence to be embedded
        :return: tensor with embeddings of all input sentences
        :param partial_save_path: file with partially processed sentences
        """
        logger.debug(f"# of all sentences: {len(sentences)}")
        embeddings = np.empty((0, 512))

        if partial_save_path is not None:
            partial_save_path = Path(partial_save_path).with_suffix(".npy")
            if partial_save_path.exists():
                embeddings = np.load(partial_save_path.as_posix())
                sentences = sentences[embeddings.shape[0] :]
                logger.debug(f"# of already embedded sentences: {embeddings.shape[0]}.")

        n_chunks = np.ceil(len(sentences) / max_chunk_size)
        logger.debug(f"# of all chunks: {n_chunks} embedding iterations.")

        for sentences_chunk in tqdm(
            chunked(sentences, max_chunk_size),
            desc="Embed sentences in chunks",
            total=n_chunks,
        ):
            values, indices, dense_shape = process_to_ids_in_sparse_format(
                self.sentence_piece_processor, sentences_chunk
            )

            embeddings = np.append(
                embeddings,
                self.session.run(
                    self.computation,
                    feed_dict={
                        self.input_placeholder.values: values,
                        self.input_placeholder.indices: indices,
                        self.input_placeholder.dense_shape: dense_shape,
                    },
                ),
                axis=0,
            )

            if partial_save_path is not None:
                np.save(partial_save_path.as_posix(), embeddings)

        return embeddings

    def embed_sentence(self, sentence: str) -> np.ndarray:
        return self.embed_sentences([sentence])[0]


class EmbeddingSimilarity:
    def __init__(self, embedding_size: int = 512):
        self.x = tf.placeholder(tf.float32, shape=(None, embedding_size))
        self.y = tf.placeholder(tf.float32, shape=(None, embedding_size))

        normed_source_embeddings = tf.nn.l2_normalize(self.x, axis=1)
        normed_target_embedding = tf.nn.l2_normalize(self.y, axis=1)

        tf_angular_similarity = (
            1
            - tf.acos(
                tf.matmul(
                    normed_source_embeddings,
                    tf.transpose(normed_target_embedding, [1, 0]),
                )
            )
            / np.pi
        )
        self.tf_clip_angular_similarity = tf.clip_by_value(
            tf_angular_similarity, 0.0, 1.0
        )

    def compute_angular_similarity(self, x, y):
        with tf.Session() as session:
            return session.run(
                self.tf_clip_angular_similarity, feed_dict={self.x: x, self.y: y}
            )


@lru_cache(1)
def load_guse(embedding_model_path: str = None):
    return GuseEmbedder(embedding_model_path)


if __name__ == "__main__":
    load_guse()
