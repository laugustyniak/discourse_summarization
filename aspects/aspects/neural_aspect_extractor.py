import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

from embeddings.utils import get_embedding_vocab
from utilities import settings
from utilities.common_nlp import load_spacy


class NeuralAspectExtractor:

    def __init__(self):
        self.model, self.model_info = self._load_model()
        self.word_embedding_vocab = get_embedding_vocab()

    def _load_model(self):
        with open(settings.ASPECT_EXTRACTION_NEURAL_MODEL_INFO.as_posix(), 'rb') as f:
            model_info = pickle.load(f)
        model = load_model(
            settings.ASPECT_EXTRACTION_NEURAL_MODEL.as_posix(),
            custom_objects={
                'CRF': CRF,
                'loss': crf_loss
            }
        )
        return model, model_info

    def predict(self, text):
        text_padded = self._get_padded_text(text)
        prediction = self.model.predict([text_padded])
        return prediction[0]

    def _get_padded_text(self, text):
        nlp = load_spacy()
        return pad_sequences(
            [[
                self.word_embedding_vocab[token.text]
                for token
                in nlp.make_doc(text)
            ]],
            maxlen=self.model_info['sentence_len'],
            padding='post'
        )


if __name__ == '__main__':
    nae = NeuralAspectExtractor()
    nae.predict('this phone has great screen')
    pass
