import pickle
from pathlib import Path

import spacy
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

nlp = spacy.load('en_core_web_sm')

MODEL_NAME = 'model-72eb2ef'
ASPECT_EXTRACTION_NEURAL_MODEL = Path('models') / f'{MODEL_NAME}.h5'
ASPECT_EXTRACTION_NEURAL_MODEL_INFO = Path('models') / f'{MODEL_NAME}.info'


class NeuralAspectExtractor:
    TAG_VOCAB = {
        'O': 1,
        'I-aspect': 2
    }
    OOV_WORD_ID = 1

    def __init__(self):
        self.model, self.model_info = self._load_model()
        self.word_embedding_vocab = self.model_info['word_vocab']
        # uncomment after adding char embeddings
        # self.char_embedding_vocab = self.model_info['char_vocab']

    @staticmethod
    def _load_model():
        # check of models are downloaded, if not download them
        with open(ASPECT_EXTRACTION_NEURAL_MODEL_INFO.as_posix(), 'rb') as f:
            model_info = pickle.load(f)
        model = load_model(
            ASPECT_EXTRACTION_NEURAL_MODEL.as_posix(),
            custom_objects={
                'CRF': CRF,
                'crf_loss': crf_loss,
                'crf_accuracy': crf_accuracy
            }
        )
        return model, model_info

    def extract(self, text):
        text_padded = self._get_padding(
            text, self.word_embedding_vocab, self.model_info['sentence_len'])
        prediction = self.model.predict(text_padded)

        prediction = prediction[0][:, 2]

        all_aspects = []
        aspects = []
        for idx, token in enumerate(nlp.make_doc(text)):
            if idx < self.model_info['sentence_len'] and prediction[idx]:
                aspects.append(token.text)
            else:
                if aspects:
                    all_aspects.append(' '.join(aspects))
                aspects = []
        if aspects:
            all_aspects.append(' '.join(aspects))
        return all_aspects

    def _get_padding(self, text, vocab, max_padding):
        return pad_sequences(
            [[
                vocab[token.text] if token.text in vocab else self.OOV_WORD_ID
                for token
                in nlp.make_doc(text)
            ]],
            maxlen=max_padding,
            padding='post'
        )


def get_padding_for_tokens(tokens, vocab, max_padding):
    return pad_sequences(
        [[
            vocab[token] if token in vocab else NeuralAspectExtractor.OOV_WORD_ID
            for token
            in tokens
        ]],
        maxlen=max_padding,
        padding='post'
    )
