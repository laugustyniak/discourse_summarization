import logging
from json import JSONDecodeError
from typing import Union, Tuple, List

import nltk

from rst.edu_tree_mapper import EDUTreeMapper
from rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from rst.parser_client import RSTParserClient
from utilities import settings


def extract_discourse_tree(document: str) -> Union[nltk.Tree, None]:
    parser = RSTParserClient()
    try:
        return nltk.tree.Tree.fromstring(
            parser.parse(document),
            leaf_pattern=settings.DISCOURSE_TREE_LEAF_PATTERN,
            remove_empty_top_bracketing=True
        )
    except (ValueError, JSONDecodeError) as e:
        logging.info(f'Document with errors: {document}. Error: {str(e)}')
        return None


def extract_discourse_tree_with_ids_only(discourse_tree: nltk.Tree) -> Tuple[nltk.Tree, List[str]]:
    edu_tree_preprocessor = EDUTreeMapper()
    edu_tree_preprocessor.process_tree(discourse_tree)
    return discourse_tree, edu_tree_preprocessor.edus


def extract_rules(discourse_tree: nltk.Tree) -> List:
    rules_extractor = EDUTreeRulesExtractor(tree=discourse_tree)
    return rules_extractor.extract()
