import pickle
from functools import lru_cache
from pathlib import Path

import click
import numpy as np
from keras import Input, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import (
    Embedding,
    Dropout,
    TimeDistributed,
    Bidirectional,
    LSTM,
    Dense,
)
from keras.utils import to_categorical
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm

from aspects.aspects.neural_aspect_extractor import get_padding_for_tokens, NeuralAspectExtractor
from embeddings.utils import get_word_embedding_vocab
from utilities import settings
from utilities.git_utils import get_git_revision_short_hash


@click.command()
@click.option('--dataset-path', required=False, type=Path, default=None, help='Path to the training dataset')
@click.option('--word-embedding-path', required=False, type=Path, default=None, help='Path to the embeddings')
@click.option('--word-embedding-dim', required=False, type=int, default=300, help='Len of word embedding vectors')
# @click.option('--char-embedding-dim', required=False, type=int, default=25, help='Len of character embedding vectors')
@click.option('--epochs', required=False, default=15, help='Number of epochs to calculate')
@click.option('--tag-number', required=False, default=2, help='Number of column with tag to classify')
@click.option('--sentence-length', required=False, type=int, default=30)
@click.option('--word-length', required=False, type=int, default=20)
@click.option('--batch-size', required=False, type=int, default=32)
@click.option('--dropout', required=False, type=float, default=0.5)
def train_aspect_extractor(
        dataset_path: Path,
        word_embedding_path: Path,
        word_embedding_dim: int,
        # char_embedding_dim: int,
        epochs: int,
        tag_number: int,
        sentence_length: int,
        word_length: int,
        batch_size: int,
        dropout: float,
):
    if dataset_path is None:
        dataset_path = settings.ASPECT_EXTRACTION_TRAIN_DATASET

    if word_embedding_path is None:
        word_embedding_path = settings.WORD_EMBEDDING_GLOVE_42B

    click.echo('Word embedding: ' + word_embedding_path.as_posix())

    logs_path = settings.ASPECT_EXTRACTION_NEURAL_MODEL_PATH / 'logs'
    logs_path.mkdir(exist_ok=True, parents=True)

    checkpoints_path = settings.ASPECT_EXTRACTION_NEURAL_MODEL_PATH / 'checkpoints'
    checkpoints_path.mkdir(exist_ok=True, parents=True)

    model_name = f'model-{get_git_revision_short_hash()}'
    model_path = settings.ASPECT_EXTRACTION_NEURAL_MODEL_PATH / model_name

    dataset_path = dataset_path.as_posix()

    dataset = SequentialTaggingDataset(
        dataset_file_path=dataset_path,
        tag_fields=tag_number
    )
    texts, tags = list(zip(*dataset.read()))
    word_embedding_vocab = get_word_embedding_vocab()

    x_train = np.vstack([get_padding_for_tokens(tokens, word_embedding_vocab, sentence_length) for tokens in texts])

    num_y_labels = 3
    vocabulary_size = len(word_embedding_vocab) + 2
    # char_vocabulary_size = dataset.char_vocab_size + 1

    y_train = pad_sequences(
        [
            [
                NeuralAspectExtractor.TAG_VOCAB[token]
                for token
                in doc_tags
            ]
            for doc_tags
            in tags
        ],
        maxlen=sentence_length,
        padding='post'
    )
    y_train = to_categorical(y_train, num_y_labels)

    aspect_model = get_aspect_model(
        sentence_length=sentence_length,
        target_label_dims=num_y_labels,
        tagger_lstm_dims=word_embedding_dim,
        # tagger_lstm_dims=word_embedding_dim + char_embedding_dim,
        tagger_fc_dims=word_embedding_dim,
        # tagger_fc_dims=word_embedding_dim + char_embedding_dim,
        dropout=dropout,
        word_embedding_model_path=word_embedding_path.as_posix(),
    )

    # Set callback functions to early stop training and save the best model so far
    tensorboard_path = (logs_path / ('tensorboard-' + model_name)).as_posix()
    print('Tensorboard: ' + tensorboard_path)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_path)

    checkpoint_callback = ModelCheckpoint(
        (checkpoints_path / model_name).with_suffix('.hdf5').as_posix(),
        monitor='crf_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    # early_stopping_callback = EarlyStopping(
    #     monitor='crf_accuracy', mode='min', verbose=1, patience=10, min_delta=0.001)

    aspect_model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

    aspect_model.save(model_path.with_suffix('.h5').as_posix())

    # saving model info
    with open(model_path.with_suffix('.info').as_posix(), 'wb') as fp:
        info = {
            'sentence_len': sentence_length,
            'word_len': word_length,
            'num_of_labels': num_y_labels,
            # 'labels_id_to_word': {v: k for k, v in dataset.y_labels.vocab.items()},
            'epoch': epochs,
            'word_vocab': word_embedding_vocab,
            'vocab_size': vocabulary_size,
            # 'char_vocab_size': char_vocabulary_size,
            # 'char_vocab': word_embedding_vocab,
            'word_embedding_dims': word_embedding_dim,
            # 'char_embedding_dims': char_embedding_dim,
            # 'word_lstm_dims': char_embedding_dim,
            # 'tagger_lstm_dims': word_embedding_dim + char_embedding_dim,
            'tagger_lstm_dims': word_embedding_dim,
            'dropout': dropout,
            'external_embedding_model': word_embedding_path,
            'train_file': dataset_path,
            'y_train': y_train
        }
        pickle.dump(info, fp)


def get_aspect_model(
        sentence_length: int,
        target_label_dims: int,
        tagger_lstm_dims: int,
        tagger_fc_dims: int,
        dropout: float,
        word_embedding_model_path: str = None,
):
    all_inputs = []

    # build word input
    words_input = Input(shape=(sentence_length,), name='words_input')
    all_inputs.append(words_input)

    embedding_layer = create_embedding_layer(word_embedding_model_path)

    word_embeddings = embedding_layer(words_input)
    word_embeddings = Dropout(dropout)(word_embeddings)
    # TODO: add character embedding
    features = word_embeddings

    lstm_layers = Bidirectional(LSTM(tagger_lstm_dims, return_sequences=True))(features)
    lstm_layers = Dropout(dropout)(lstm_layers)
    after_lstm_hidden = TimeDistributed(Dense(tagger_fc_dims))(lstm_layers)

    crf = CRF(target_label_dims, sparse_target=False)
    predictions = crf(after_lstm_hidden)
    # compile the model
    model = Model(inputs=all_inputs, outputs=predictions)
    model.compile(
        loss=crf_loss,
        optimizer='adam',
        metrics=[crf_accuracy]
    )

    return model


def create_embedding_layer(word_embedding_path):
    # load and prepare external word embedding
    word_embedding, word_embedding_vector_size = load_word_embeddings(word_embedding_path)
    embedding_matrix = np.zeros((len(word_embedding), word_embedding_vector_size))
    for i, word in enumerate(word_embedding):
        embedding_vector = word_embedding.get(word.lower())  # TODO: lower?
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    return Embedding(
        len(word_embedding),
        word_embedding_vector_size,
        weights=[embedding_matrix],
        input_length=30,
        trainable=False
    )


@lru_cache(maxsize=1)
def load_word_embeddings(file_path):
    """
    Loads a word embedding model text file into a word(str) to numpy vector dictionary

    Args:
        file_path (str): path to model file

    Returns:
        list: a dictionary of numpy.ndarray vectors
        int: detected word embedding vector size
    """
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


class SequentialTaggingDataset:

    def __init__(self, dataset_file_path, tag_fields):

        self.dataset_file_path = dataset_file_path
        self.tag_fields = tag_fields

    def read(self):
        with open(self.dataset_file_path, encoding='utf-8') as fp:
            file_lines = fp.readlines()
            file_lines = [d.strip() for d in file_lines]
            file_lines = [d for d in file_lines if 'DOCSTART' not in d]
            sentences = self._split_into_sentences(file_lines)
            parsed_sentences = [self._parse_sentence(sentence) for sentence in sentences if len(sentence) > 0]
        return parsed_sentences

    def _parse_sentence(self, sentence):
        tokens = []
        tags = []
        for line in sentence:
            fields = line.split()
            assert len(fields) >= self.tag_fields, 'tag field exceeds number of fields'
            if 'CD' in fields[1]:
                tokens.append('0')
            else:
                tokens.append(fields[0])
            tags.append(fields[self.tag_fields - 1])
        return tokens, tags

    @staticmethod
    def _split_into_sentences(file_lines):
        sents = []
        s = []
        for line in file_lines:
            line = line.strip()
            if not line:
                sents.append(s)
                s = []
                continue
            s.append(line)
        sents.append(s)
        return sents


if __name__ == '__main__':
    train_aspect_extractor()
