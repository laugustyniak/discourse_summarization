from pathlib import Path
from tqdm import tqdm


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
