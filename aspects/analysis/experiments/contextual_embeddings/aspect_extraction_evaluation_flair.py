from pathlib import Path

import click
from flair.datasets import ColumnCorpus
from flair.embeddings import (
    StackedEmbeddings,
    WordEmbeddings,
    BertEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from tqdm import tqdm

from aspects.utilities.settings import DEFAULT_OUTPUT_PATH, SEMEVAL_DATASETS_2014

# TODO: remove after experiements
RESULTS_PATH = DEFAULT_OUTPUT_PATH / 'reviews_Cell_Phones_and_Accessories-50000-docs' / 'our'


@click.command()
@click.option('--input-dir', type=click.Path(), default=SEMEVAL_DATASETS_2014 / 'poria')
@click.option('--output-dir', type=click.Path(), default=RESULTS_PATH)
def run_experiments(input_dir: Path, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # retrieve corpus using column format, data folder and the names of the train, dev and test files
    corpus = ColumnCorpus(
        input_dir,
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

    all_embedding_to_test = {
        # 'glove+aspects': [
        #     WordEmbeddings('glove'),
        #     WordEmbeddings(
        #         (output_dir / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.word2vec_format.bin').as_posix()
        #     ),
        # ],
        # 'glove': [
        #     WordEmbeddings('glove'),
        # ],
        # 'charlmembedding': [
        #     FlairEmbeddings('news-forward'),
        #     FlairEmbeddings('news-backward'),
        # ],
        # 'glove-simple-char': [
        #     WordEmbeddings('glove'),
        #     CharacterEmbeddings(),
        # ],
        'bert+aspects': [
            BertEmbeddings('bert-large-cased'),
            WordEmbeddings(
                (output_dir / 'aspect_2_aspect_graph-en_core_web_lg.en_core_web_lg.word2vec_format.bin').as_posix()
            )
        ],
        'bert': [
            BertEmbeddings('bert-large-cased'),
        ],
        # 'elmo': [
        #     ELMoEmbeddings('original')
        # ]
    }

    for name, embeddings_to_stack in tqdm(
            all_embedding_to_test.items(),
            desc='Different embeddings stacked',
            total=len(all_embedding_to_test)
    ):
        results_folder = Path(DEFAULT_OUTPUT_PATH / f'sequence-tagging/aspects/laptops-{name}')
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings_to_stack)

        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=True
        )

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        # 7. start training
        trainer.train(
            results_folder.as_posix(),
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=150
        )

        # 8. plot training curves (optional)
        plotter = Plotter()
        plotter.plot_training_curves(results_folder / 'loss.tsv')
        plotter.plot_weights(results_folder / 'weights.txt')


if __name__ == '__main__':
    run_experiments()
