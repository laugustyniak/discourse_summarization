import re

import spacy

from document.dependency import Dependency
from document.sentence import Sentence
from document.token import Token
from prep import prep_utils
from prep.syntax_parser import SyntaxParser
from trees.lexicalized_tree import LexicalizedTree


class Preprocesser:

    def __init__(self):
        self.syntax_parser = None

        try:
            self.syntax_parser = SyntaxParser()
        except Exception as e:
            raise Exception(str(e) + 'Please check paths.py file and ROOT PATH if it points to the right directory')

        self.max_sentence_len = 100
        self.nlp = spacy.load('en')

    def parse_single_sentence(self, raw_text):
        return self.syntax_parser.parse_sentence(raw_text)

    def process_single_sentence(self, doc, raw_text):
        sentence = Sentence(len(doc.sentences), raw_text, doc)
        parse_tree_str, deps_str = self.parse_single_sentence(raw_text)

        # TODO initialization of Lexicalized Trees
        parse = LexicalizedTree.parse(parse_tree_str, leaf_pattern='(?<=\\s)[^\)\(]+')
        sentence.set_unlexicalized_tree(parse)

        for (token_id, te) in enumerate(parse.leaves()):
            word = te
            token = Token(word, token_id + 1, sentence)
            sentence.add_token(token)

        heads = self.get_heads(sentence, deps_str.split('\n'))
        sentence.heads = heads
        sentence.set_lexicalized_tree(prep_utils.create_lexicalized_tree(parse, heads))

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

    def preprocess(self, text, doc):
        doc.sentences = []
        for sentence in self.nlp(unicode(text)).sents:
            self.process_single_sentence(doc, sentence.text)

    def unload(self):
        if self.syntax_parser:
            self.syntax_parser.unload()
