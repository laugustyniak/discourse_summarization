from pathlib import Path

from tqdm import tqdm

dataset_restaurants_test = 'Restaurants_poria-test'
dataset_restaurants_train = 'Restaurants_poria-train'

dataset_laptops_test = 'Laptops_poria-test'
dataset_laptops_train = 'Laptops_poria-train'


def get_unique_words_from_corpus(
        corpus_paths=Path(
            '/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/semeval/2014/poria').glob(
            '*conll')):
    corpus_words = {}
    for corpus_path in [Path(c) for c in corpus_paths]:
        with open(corpus_path) as corpus_file:
            corpus_words[corpus_path.stem] = set(
                line.split('\t')[0].lower()
                for line
                in tqdm(corpus_file, desc='Corpus iterator')
                if line
            )
    return corpus_words


def get_uni_and_multigram_aspects_stats(
        corpus_paths=Path(
            '/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/semeval/2014/poria').glob(
            '*conll'),
        col_number_of_aspects: int = 2,
        sep: str = '\t'
):
    aspects_ngrams = {}
    for corpus_path in [Path(c) for c in corpus_paths]:
        with open(corpus_path) as corpus_file:
            aspects = [
                line.strip().split(sep)
                for line
                in tqdm(corpus_file, desc='Corpus iterator')
                if len(line.split(sep)) > col_number_of_aspects
            ]

            previous_i = False
            n_all_aspects = 0
            n_multi_aspects = 0
            for aspect in aspects:
                if 'B' in aspect[col_number_of_aspects]:
                    n_all_aspects += 1
                    previous_i = False
                elif 'I' in aspect[col_number_of_aspects]:
                    if not previous_i:
                        n_multi_aspects += 1
                        previous_i = True
                    else:
                        pass
                else:
                    previous_i = False

            aspects_ngrams[corpus_path.stem] = round(n_multi_aspects / n_all_aspects, 4) * 100

    return aspects_ngrams


def get_aspects(
        corpus_paths=Path(
            '/home/laugustyniak/github/phd/nlp-architect/examples/aspect_extraction/semeval/2014/poria').glob(
            '*conll'),
        col_number_of_aspects: int = 2,
        sep: str = '\t'
):
    aspects = {}
    for corpus_path in [Path(c) for c in corpus_paths]:
        with open(corpus_path) as corpus_file:
            aspects[corpus_path.stem] = []

            aspects_lines = [
                line.strip().split(sep)
                for line
                in tqdm(corpus_file, desc='Corpus iterator')
                if len(line.split(sep)) > col_number_of_aspects
            ]

            previous_i = False
            aspect = ''
            for aspect_line in aspects_lines:
                if 'B' in aspect_line[col_number_of_aspects]:
                    aspect = aspect_line[0]
                    previous_i = False
                elif 'I' in aspect_line[col_number_of_aspects]:
                    aspect = f'{aspect} {aspect_line[0]}'
                    if not previous_i:
                        previous_i = True
                    else:
                        pass
                else:
                    if aspect:
                        aspects[corpus_path.stem].append(aspect.lower())
                        aspect = ''
                    previous_i = False
    return aspects


if __name__ == '__main__':
    get_aspects()
