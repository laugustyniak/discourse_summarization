from base_parser import BaseParser


class MultiSententialParser(BaseParser):
    def __init__(self, name='MultiParser', verbose=False, window_size=3):
        BaseParser.__init__(self, name, verbose, window_size)

        self.scope = False

    def add_classifier(self, classifier, name):
        if name == 'bin':
            self.bin_classifier = classifier
        elif name == 'mc':
            self.mc_classifier = classifier
        else:
            raise Exception('Unknown classifier name', name)

        print 'Added classifier', name, 'to treebuilder', self.name

    def parse_document(self, doc):
        self.clear_cache()

        doc.prepare_parsing()

        return self.parse_sequence(doc)

    def parse_sequence(self, doc):
        if len(doc.constituents) == 1:
            doc.discourse_tree = doc.constituents[0].parse_subtree
            return

        for i in range(len(doc.constituents) - 1):
            bin_score = self.classify_pair(doc, i)
            doc.constituent_scores.append(bin_score)

        seq_prob = None
        while len(doc.constituents) > 1:
            best_one = None
            max_bin_score = -20.0
            for (index, bin_score) in enumerate(doc.constituent_scores):
                if bin_score > max_bin_score:
                    best_one = index
                    max_bin_score = bin_score

            seq_prob = self.connect_stumps(best_one, doc)

        if doc.constituents:
            doc.discourse_tree = doc.constituents[0].parse_subtree
        else:
            doc.discourse_tree = u''

    def classify_pair(self, doc, i):
        max_prob = -20
        max_prob_sequence = None
        max_prob_predictions = None

        for (s, j) in self.generate_crf_sequences(doc.constituents, i, labeling=False):
            #            print 's', s
            (sequence_prob, predictions) = self.parse_single_sequence(s, labeling=False)

            if sequence_prob > max_prob:
                max_prob = sequence_prob
                max_prob_sequence = (s, j)
                max_prob_predictions = predictions

        (s_star, j_star) = max_prob_sequence
        struct_prob = max_prob_predictions[j_star]
        return struct_prob

    def relabel_stumps(self, doc, i):
        max_prob = -20
        max_prob_sequence = None
        max_prob_predictions = None

        for (s, j) in self.generate_crf_sequences(doc.constituents, i, labeling=True):
            (prob, predictions) = self.parse_single_sequence(s, labeling=True)
            if prob > max_prob:
                max_prob = prob
                max_prob_sequence = (s, j)
                max_prob_predictions = predictions

        (s_star, j_star) = max_prob_sequence

        for k in range(len(s_star)):
            c = s_star[k]

            if not c.is_leaf():
                c1 = c.left_child
                c2 = c.right_child

                predicted_label = max_prob_predictions[k]
                c.parse_subtree.node = predicted_label

                if self.verbose:
                    print 'Relabling'
                    print 'L', c1
                    print c1.parse_subtree
                    print 'R', c2
                    print c2.parse_subtree
                    print 'with predicted label', predicted_label
                    print

        return max_prob, i - j_star, len(s_star)

    def connect_stumps(self, i, doc):
        if self.verbose:
            print 'Connecting stumps[%d] and stumps[%d]' % (i, i + 1)
            print 'stumps[%d]' % i, doc.constituents[i]
            print
            print 'stumps[%d]' % (i + 1), doc.constituents[i + 1]

        L = doc.constituents[i]
        R = doc.constituents[i + 1]
        new_constituent = L.make_new_constituent('n/a', R)
        doc.constituents[i:i + 2] = [new_constituent]

        (seq_prob, start, s_len) = self.relabel_stumps(doc, i)
        for k in range(max(0, start - (self.window_size - 1) / 2 - 1), i):
            bin_score = self.classify_pair(doc, k)
            doc.constituent_scores[k] = bin_score

        for k in range(i + 2, min(len(doc.constituents) - 1, i + 2 + (self.window_size - 1) / 2 + s_len)):
            bin_score = self.classify_pair(doc, k - 1)
            doc.constituent_scores[k] = bin_score

        doc.constituent_scores[i: i + 1] = []
        return seq_prob
