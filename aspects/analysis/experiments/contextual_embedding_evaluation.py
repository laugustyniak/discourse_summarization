from pathlib import Path
from typing import List

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import StackedEmbeddings, TokenEmbeddings, WordEmbeddings, CharLMEmbeddings, CharacterEmbeddings
from flair.models import SequenceTagger
from flair.trainers import SequenceTaggerTrainer
from flair.visual.training_curves import Plotter
from tqdm import tqdm

# this is the folder in which train, test and dev files reside
DATA_FOLDER = Path('semeval/2014/poria/')

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_column_corpus(
    DATA_FOLDER,
    {0: 'text', 1: 'dep', 2: 'aspect'},
    train_file='Laptops_poria-train.conll',
    # train_file='Restaurants_poria-train.conll',
    test_file='Laptops_poria-test.conll',
    # test_file='Restaurants_poria-test.conll',
    dev_file='Laptops_poria-train.conll'
    # dev_file='Restaurants_poria-train.conll'
)

# 2. what tag do we want to predict?
tag_type = 'aspect'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

for name, embedding in tqdm({
                                'glove': [
                                    WordEmbeddings('glove'),
                                ],
                                'glove-simple-char': [
                                    WordEmbeddings('glove'),
                                    CharacterEmbeddings(),
                                ],
                                'glove-charlmembedding-simple-char': [
                                    WordEmbeddings('glove'),
                                    CharacterEmbeddings(),
                                    CharLMEmbeddings('news-forward'),
                                    CharLMEmbeddings('news-backward'),
                                ],
                                'glove-charlmembedding': [
                                    WordEmbeddings('glove'),
                                    CharLMEmbeddings('news-forward'),
                                    CharLMEmbeddings('news-backward'),
                                ],
                                'charlmembedding': [
                                    CharLMEmbeddings('news-forward'),
                                    CharLMEmbeddings('news-backward'),
                                ],

                            }.items(),
                            desc='Different embeddings stacked', total=4):
    RESULTS_FOLDER = Path(f'resources/taggers/example-aspect-laptops-{name}')
    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        CharacterEmbeddings(),

        # comment in these lines to use contextual string embeddings
        CharLMEmbeddings('news-forward'),
        CharLMEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True
    )

    trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus)

    # 7. start training
    trainer.train(
        RESULTS_FOLDER.as_posix(),
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=150
    )

    # 8. plot training curves (optional)
    plotter = Plotter()
    plotter.plot_training_curves(RESULTS_FOLDER / 'loss.tsv')
    plotter.plot_weights(RESULTS_FOLDER / 'weights.txt')
