import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from utilities import settings
from utilities.common_nlp import load_spacy

nlp = load_spacy()


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

    def _load_model(self):
        with open(settings.ASPECT_EXTRACTION_NEURAL_MODEL_INFO.as_posix(), 'rb') as f:
            model_info = pickle.load(f)
        model = load_model(
            settings.ASPECT_EXTRACTION_NEURAL_MODEL.as_posix(),
            custom_objects={
                'CRF': CRF,
                'crf_loss': crf_loss,
                'crf_accuracy': crf_accuracy
            }
        )
        return model, model_info

    def predict(self, text):
        text_padded = self._get_padding(text, self.word_embedding_vocab, self.model_info['sentence_len'])
        prediction = self.model.predict(text_padded)
        return prediction[0]

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


if __name__ == '__main__':
    nae = NeuralAspectExtractor()
    nae.predict('has a really good screen quality')
    pass
