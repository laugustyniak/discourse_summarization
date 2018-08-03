from collections import namedtuple

from tqdm import tqdm

from aspects.preprocessing.transform_formats import TextTag

Sentence = namedtuple('Sentence', ['words', 'tags'])


class Conll(object):

    def __init__(self, file_path: str, n_tag_fields: int = 2):
        self.file_path = file_path
        self.n_tag_fields = n_tag_fields

    def read_file(self):
        with open(self.file_path, encoding='utf-8') as fp:
            data = fp.readlines()
            data = [d.strip() for d in data]
            data = [d for d in data if 'DOCSTART' not in d]
            sentences = self._split_into_sentences(data)
            parsed_sentences = [self._parse_sentence(s) for s in sentences if len(s) > 0]
        return parsed_sentences

    def _parse_sentence(self, sentence):
        tokens = []
        tags = []
        for line in sentence:
            fields = line.split()
            assert len(fields) >= self.n_tag_fields, 'tag field exceeds number of fields'
            if 'CD' in fields[1]:
                tokens.append('0')
            else:
                tokens.append(fields[0])
            tags.append(fields[self.n_tag_fields - 1])
        return Sentence(tokens, tags)

    @staticmethod
    def _split_into_sentences(file_lines):
        sents = []
        s = []
        for line in file_lines:
            line = line.strip()
            if len(line) == 0:
                sents.append(s)
                s = []
                continue
            s.append(line)
        sents.append(s)
        return sents

    def extract_words_and_tags(self, sentences):
        for sentence in tqdm(sentences):
            text_tag = list(filter(lambda t: t[1] != 'O', zip(sentence.words, sentence.tags)))

            if len(text_tag) == 1:
                yield TextTag(text=text_tag[0][0], tag=text_tag[0][1].replace('B-', ''))
            elif len(text_tag) > 1:
                text_prev, tag_prev = text_tag[0]
                for text_next, tag_next in text_tag[1:]:
                    if tag_next.startswith('B-'):
                        yield TextTag(text_prev, tag_next.replace('B-', ''))
                        text_prev = text_next
                    else:
                        text_prev += ' ' + text_next

                if tag_next.startswith('I-'):
                    yield TextTag(text_prev, tag_next.replace('I-', ''))
            else:
                pass
