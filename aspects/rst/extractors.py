from typing import Union, Tuple, List

import nltk

from aspects.rst.edu_tree_mapper import EDUTreeMapper
from aspects.rst.edu_tree_rules_extractor import EDUTreeRulesExtractor
from aspects.rst.parser_client import RSTParserClient
from aspects.utilities import settings


def extract_discourse_tree(document: str) -> Union[nltk.Tree, None]:
    parser = RSTParserClient()
    parse_tree_str = parser.parse(document)
    return nltk.tree.Tree.fromstring(
        parse_tree_str,
        leaf_pattern=settings.DISCOURSE_TREE_LEAF_PATTERN,
        remove_empty_top_bracketing=True,
    )


def extract_discourse_tree_with_ids_only(
    discourse_tree: nltk.Tree,
) -> Tuple[nltk.Tree, List[str]]:
    edu_tree_preprocessor = EDUTreeMapper()
    edu_tree_preprocessor.process_tree(discourse_tree)
    return discourse_tree, edu_tree_preprocessor.edus


def extract_rules(discourse_tree: nltk.Tree) -> List:
    rules_extractor = EDUTreeRulesExtractor(tree=discourse_tree)
    return rules_extractor.extract()
