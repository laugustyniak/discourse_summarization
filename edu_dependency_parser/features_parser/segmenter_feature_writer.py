from nltk.tree import Tree, _child_names

from utils import utils, rst_lib


class SegmenterFeatureWriter:
    def __init__(self):
        self.cached_subtrees = {}
        pass

    def write_token_identity_features(self, token, unit, position):
        self.features.add('PoS=%s_Unit%d@%d' % (token.get_PoS_tag(), unit, position))
        self.features.add('Word=%s_Unit%d@%d' % (token.word.lower(), unit, position))

        if token.is_sentence_begin():
            self.features.add('Is_Sentence_Begin_Unit%d@%d' % (unit, position))

        if token.is_sentence_end():
            self.features.add('Is_Sentence_End_Unit%d@%d' % (unit, position))

    def write_unit_token_identity_features(self, token, unit, position):
        treepos = token.get_treepos()
        tree = token.sentence.parse_tree
        while len(treepos) > 0:
            if unit == 1:
                pos = len(tree[treepos[: -1]]) - 1
            else:
                pos = 0

            if treepos[-1] != pos:
                break

            treepos = treepos[: -1]

        ancestor_subtree = tree[treepos]
        self.features.add('Largest_Subtree_Top_Tag=%s_Unit=%d@%d' % (ancestor_subtree.node, unit, position))
        self.features.add('Largest_Subtree_Depth=%s_Unit=%d@%d' % (len(treepos), unit, position))

        if len(treepos) < len(token.get_treepos()):
            production = '%s->%s' % (ancestor_subtree.node, '#'.join(str(x) for x in _child_names(ancestor_subtree)))
            self.features.add('Largest_Subtree_Production=%s_Unit=%d@%d' % (production, unit, position))

    def write_token_pair_features(self, token1, token2, position):
        l_treepos = token1.get_treepos()
        r_treepos = token2.get_treepos()

        common_ancestor_pos = rst_lib.common_ancestor(l_treepos, r_treepos)

        dist_ancestor_l = len(l_treepos) - len(common_ancestor_pos)
        dist_ancestor_r = len(r_treepos) - len(common_ancestor_pos)

        if dist_ancestor_l == 0:  # L >> R
            self.features.add('L_Dominates_R@%d' % position)
        elif dist_ancestor_r == 0:  # R >> L
            self.features.add('R_Dominates_L@%d' % position)

    def write_global_features(self, token, offset2neighbouring_boundaries, unit, position):
        """ Find the left and right neighbouring boundaries """

        (l_boundary, r_boundary) = offset2neighbouring_boundaries[token.id - 1]

        if l_boundary is not None:
            start = l_boundary
            self.features.add('L_Boundary_Word=%s_Unit%d@%d' % (
                token.sentence.tokens[l_boundary].word.lower(), unit, position))
            self.features.add('L_Boundary_POS=%s_Unit%d@%d' % (
                token.sentence.tokens[l_boundary].get_PoS_tag(), unit, position))
            self.features.add(
                'Distance_to_L_Neighbouring_Boundary=%d_Unit%d@%d' % (token.id - l_boundary, unit, position))

            self.write_global_features_for_span((l_boundary, token.id), token, unit, position)
        else:
            start = 0
            self.features.add('First_Boundary@%d' % (position))

        if r_boundary is not None:
            end = r_boundary

            self.features.add('R_Boundary_Word=%s_Unit%d@%d' % (
                token.sentence.tokens[r_boundary - 1].word.lower(), unit, position))
            self.features.add('R_Boundary_POS=%s_Unit%d@%d' % (
                token.sentence.tokens[r_boundary - 1].get_PoS_tag(), unit, position))
            self.features.add('Distance_to_R_Neighbouring_Boundary=%d_Unit%d@%d' % (
                    r_boundary - token.id + 1, unit, position))

            self.write_global_features_for_span((token.id - 1, r_boundary), token, unit, position)
        else:
            end = len(token.sentence.tokens)
            self.features.add('Last_Boundary@%d' % (position))

        self.write_global_features_for_span((start, end), token, -unit, position)

    def write_global_features_for_span(self, span, token, unit, position):
        (start, end) = span

        tree = token.sentence.parse_tree

        if (start, end) in self.cached_subtrees:
            subtrees = self.cached_subtrees[(start, end)]
        else:
            subtrees = utils.get_syntactic_subtrees(tree, start, end)
            self.cached_subtrees[(start, end)] = subtrees

        self.features.add('Subtrees_to_Neighbouring_Boundary=%d_Unit%d@%d' % (len(subtrees), unit, position))
        subtree_top_tags = []
        for subtree in subtrees:
            subtree_top_tags.append(subtree.node)

        ancestor_treepos = tree.treeposition_spanning_leaves(start, end)

        ancestor_subtree = tree[ancestor_treepos]
        if not isinstance(ancestor_subtree, Tree):
            ancestor_subtree = tree[ancestor_treepos[: -1]]
        # print ancestor_subtree

        self.features.add(
            'Ancestor_Subtree_Tag_Neighbouring_Boundary=%s_Unit%d@%d' % (ancestor_subtree.node, unit, position))
        production = '%s->%s' % (ancestor_subtree.node, '#'.join(subtree_top_tags))
        self.features.add('Ancestor_Subtree_Production_Neighbouring_Boundary=%s_Unit%d@%d' %
                          (production, unit, position))

    def write_features(self, tokens, offset2neighbouring_boundaries=None):
        self.features = set()

        for i in range(len(tokens) - 1):
            token1 = tokens[i]
            token2 = tokens[i + 1]

            if token1 and token2:
                assert token1.sentence == token2.sentence

                self.write_token_identity_features(token1, 1, i)
                self.write_token_identity_features(token2, 2, i)

                if offset2neighbouring_boundaries:
                    self.write_global_features(token1, offset2neighbouring_boundaries, 1, i)
                    self.write_global_features(token2, offset2neighbouring_boundaries, 2, i)

                self.write_unit_token_identity_features(token1, 1, i)
                self.write_unit_token_identity_features(token2, 2, i)

                self.write_token_pair_features(token1, token2, i)

        return self.features
