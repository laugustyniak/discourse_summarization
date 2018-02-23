import re

import prep_utils
from aspects.utilities.nlp import load_spacy
from document.dependency import Dependency
from document.sentence import Sentence
from document.token import Token
from syntax_parser import SyntaxParser
from trees.lexicalized_tree import LexicalizedTree

nlp = load_spacy()


class Preprocesser:
    def __init__(self):
        self.syntax_parser = None

        try:
            self.syntax_parser = SyntaxParser()
        except Exception, e:
            raise e

        self.max_sentence_len = 100

    def heuristic_sentence_splitting(self, raw_sent):
        if len(raw_sent) == 0:
            return []

        if len(raw_sent.split()) <= self.max_sentence_len:
            return [raw_sent]

        i = len(raw_sent) / 2
        j = i
        k = i + 1
        boundaries = [';', ':', '!', '?']

        results = []
        while j > 0 and k < len(raw_sent) - 1:
            if raw_sent[j] in boundaries:
                l_sent = raw_sent[: j + 1]
                r_sent = raw_sent[j + 1:].strip()

                if len(l_sent.split()) > 1 and len(r_sent.split()) > 1:
                    results.extend(self.heuristic_sentence_splitting(l_sent))
                    results.extend(self.heuristic_sentence_splitting(r_sent))
                    return results
                else:
                    j -= 1
                    k += 1
            elif raw_sent[k] in boundaries:
                l_sent = raw_sent[: k + 1]
                r_sent = raw_sent[k + 1:].strip()

                if len(l_sent.split()) > 1 and len(r_sent.split()) > 1:
                    results.extend(self.heuristic_sentence_splitting(l_sent))
                    results.extend(self.heuristic_sentence_splitting(r_sent))
                    return results
                else:
                    j -= 1
                    k += 1
            else:
                j -= 1
                k += 1

        if len(results) == 0:
            return [raw_sent]

    def parse_single_sentence(self, raw_text):
        return self.syntax_parser.parse_sentence(raw_text)

    def process_single_sentence(self, doc, raw_text, end_of_para):
        sentence = Sentence(len(doc.sentences), raw_text + ('<s>' if not end_of_para else '<P>'), doc)
        parse_tree_str, deps_str = self.parse_single_sentence(raw_text)

        parsed_tree = LexicalizedTree.parse(parse_tree_str, leaf_pattern='(?<=\\s)[^\)\(]+')
        sentence.set_unlexicalized_tree(parsed_tree)

        for (token_id, te) in enumerate(parsed_tree.leaves()):
            word = te
            token = Token(word, token_id + 1, sentence)
            sentence.add_token(token)

        heads = self.get_heads(sentence, deps_str.split('\n'))
        sentence.heads = heads
        sentence.set_lexicalized_tree(prep_utils.create_lexicalized_tree(parsed_tree, heads))

        doc.add_sentence(sentence)

    def get_heads(self, sentence, dep_elems):
        heads = []
        for token in sentence.tokens:
            heads.append([token.word, token.get_PoS_tag(), 0])

        for dep_e in dep_elems:
            m = re.match('(.+?)\((.+?)-(\d+?), (.+?)-(\d+?)\)', dep_e)
            if m:
                relation = m.group(1)
                gov_id = int(m.group(3))
                dep_id = int(m.group(5))

                heads[dep_id - 1][2] = gov_id
                sentence.add_dependency(Dependency(gov_id, dep_id, relation))

        return heads

    def sentence_splitting(self, text, doc):
        doc.sentences = []

        seg_sents = []
        sentences = [sent.string.strip() for sent in nlp(text).sents]
        n_senteces = len(sentences)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > self.max_sentence_len:
                chunked_raw_sents = self.heuristic_sentence_splitting(sentence)
                if len(chunked_raw_sents) == 1:
                    continue

                for (j, sent) in enumerate(chunked_raw_sents):
                    seg_sents.append((sent, i == n_senteces - 1 and j == len(chunked_raw_sents)))
            else:
                seg_sents.append((sentence, i == n_senteces - 1))

        for (i, (raw_text, end_of_para)) in enumerate(seg_sents):
            self.process_single_sentence(doc, raw_text, end_of_para)

    def preprocess(self, text, doc):
        self.sentence_splitting(text, doc)

    def unload(self):
        if self.syntax_parser:
            self.syntax_parser.unload()
