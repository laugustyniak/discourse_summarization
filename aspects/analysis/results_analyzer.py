# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from sklearn import metrics


class ResultsAnalyzer(object):
    def __init__(self):
        self.y_gold = []
        self.y_pred = []

    def __kappa(self, y_pred, y_gold):

        a = 0
        b = 0
        c = 0
        d = 0

        for i in range(0, len(y_pred)):
            if y_pred[i] == 1 and y_gold[i] == 1:
                a += 1
            elif y_pred[i] == 1 and y_gold[i] == 0:
                b += 1
            elif y_pred[i] == 0 and y_gold[i] == 1:
                c += 1
            else:
                d += 1

        tot = a + b + c + d
        pa = float(a + d) / tot
        p_a1 = float(a + b) / tot
        p_a2 = 1.0 - p_a1
        p_b1 = float(a + c) / tot
        p_b2 = 1.0 - p_b1
        pe = p_a1 * p_b1 + p_a2 * p_b2

        if pe == 1:
            return 1
        else:
            return (pa - pe) / (1.0 - pe)

    def __f1(self, y_pred, y_gold):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(0, len(y_pred)):
            if y_pred[i] == 1 and y_gold[i] == 1:
                tp += 1
            elif y_pred[i] == 1 and y_gold[i] == 0:
                fp += 1
            elif y_pred[i] == 0 and y_gold[i] == 1:
                fn += 1
            else:
                tn += 1

        return 2 * tp / float(2 * tp + fp + fn)

    def _convert_to_binary_vector(self, predicted_aspects, gold_standard):
        y_gold = []
        y_pred = []

        if len(predicted_aspects) == 0 and len(gold_standard) == 0:
            y_gold += [0]
            y_pred += [0]

        else:

            for aspect in predicted_aspects:

                if aspect in gold_standard:
                    y_gold += [1]
                    y_pred += [1]

                    gold_standard.remove(aspect)

                else:
                    y_gold += [0]
                    y_pred += [1]

            # fixme why aspect not defined?
            for aspect in gold_standard:
                y_gold += [1]
                y_pred += [0]

        return y_pred, y_gold

    def analyze(self, documents_aspects, gold_standard):

        y_pred, y_gold = self._convert_to_binary_vector(documents_aspects,
                                                        gold_standard)

        self.y_gold += y_gold
        self.y_pred += y_pred

    def get_analysis_results(self):

        f1 = self.__f1(self.y_gold, self.y_pred)
        hamming = metrics.hamming_loss(self.y_gold, self.y_pred)
        kappa = self.__kappa(self.y_pred, self.y_gold)

        return [f1, hamming, kappa]
