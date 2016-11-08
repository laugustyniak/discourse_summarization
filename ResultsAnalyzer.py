# -*- coding: utf-8 -*-
# author: Krzysztof xaru Rajda

from sklearn import metrics

class ResultsAnalyzer():
    
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
        Pa  = float(a + d)/tot
        PA1 = float(a + b)/tot
        PA2 = 1.0- PA1
        PB1 = float(a + c) /tot
        PB2 = 1.0 -PB1
        Pe  = PA1 *PB1 + PA2*PB2
        
        if Pe == 1:
            return 1
        else:
            return (Pa -Pe)/ (1.0 -Pe)
        
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
        
        return 2*tp / float(2*tp + fp + fn)
        
    def __convertToBinaryVector(self, predictedAspects, goldStandard):
        y_gold = []
        y_pred = []
        
        if len(predictedAspects) == 0 and len(goldStandard) == 0:
            y_gold += [0]
            y_pred += [0]
            
        else:
        
            for aspect in predictedAspects:
                
                if aspect in goldStandard:
                    y_gold += [1]
                    y_pred += [1]
                    
                    goldStandard.remove(aspect)            
                    
                else:
                    y_gold += [0]
                    y_pred += [1]
                    
            for aspect in goldStandard:
                y_gold += [1]
                y_pred += [0]
    
        return y_pred, y_gold
        
    def analyze(self, documentsAspects, goldStandard):
        
        y_pred, y_gold = self.__convertToBinaryVector(documentsAspects, goldStandard)
        
        self.y_gold += y_gold
        self.y_pred += y_pred
        
    def getAnalyzisResults(self):
        
        f1 = self.__f1(self.y_gold, self.y_pred)
        hamming = metrics.hamming_loss(self.y_gold, self.y_pred)
        kappa = self.__kappa(self.y_pred, self.y_gold)
        
        return [f1, hamming, kappa]
        