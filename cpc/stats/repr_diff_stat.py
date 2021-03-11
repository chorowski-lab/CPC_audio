

import cpc.stats.empty_stat as statTempl
import torch
import math
import os
from copy import deepcopy
import matplotlib.pyplot as plt

def euclideanDist(vecs1, vecs2):
    return torch.sqrt(torch.square(vecs1).sum(1) + torch.square(vecs2).sum(1) - (2*vecs1*vecs2).sum(1))

def euclideanDistSq(vecs1, vecs2):
    return torch.square(vecs1).sum(1) + torch.square(vecs2).sum(1) - (2*vecs1*vecs2).sum(1)

def cosineDist(vecs1, vecs2):
    cosSim = (vecs1*vecs2).sum(1) / (torch.sqrt(torch.square(vecs1).sum(1)) * torch.sqrt(torch.square(vecs2).sum(1)))
    return -cosSim + 1.

def cosineCorr(vecs1, vecs2):
    cosSim = (vecs1*vecs2).sum(1) / (torch.sqrt(torch.square(vecs1).sum(1)) * torch.sqrt(torch.square(vecs2).sum(1)))
    return torch.abs(cosSim)

class ReprDiffStat(statTempl.Stat):

    def __init__(self, metric, reprType, stepSize, histDir):
        super().__init__()
        assert metric in ('cosine', 'euclid', 'euclidsq', 'coscorr')
        assert reprType in ('repr', 'ctx')
        self.metric = metric
        self.reprType = reprType
        self.stepSize = stepSize
        self.histDir = histDir
        if not os.path.exists(self.histDir):
            os.makedirs(self.histDir)

    @staticmethod
    def convertArgsFromStrings(metric, reprType, stepSize, histDir):
        return (metric, reprType, float(stepSize), histDir)

    def computeForBatch(self, batch):
        reprData = batch[self.reprType]
        reprData1 = reprData[:,1:].contiguous().view(-1, reprData.shape[2])
        reprData2 = reprData[:,:-1].contiguous().view(-1, reprData.shape[2])
        if self.metric == 'euclid':
            distances = euclideanDist(reprData1, reprData2)
        elif self.metric == 'euclidsq':
            distances = euclideanDistSq(reprData1, reprData2)
        elif self.metric == 'cosine':
            distances = cosineDist(reprData1, reprData2)
        elif self.metric == 'coscorr':
            distances = cosineCorr(reprData1, reprData2)
        distances = torch.div(distances, self.stepSize)  #, rounding_mode='floor')
        occurences = {}
        l = 0
        for d in distances:
            if math.isnan(d):
                continue
            l += 1
            df = math.floor(d) * self.stepSize
            if df in occurences:
                occurences[df] = occurences[df] + 1
            else:
                occurences[df] = 1
        return {
            'hist': occurences,
            'sum': l
        }

    def mergeStatResults(self, prev, current):
        merged = {}
        merged['sum'] = prev['sum'] + current['sum']
        currentHist = current['hist']
        mergedHist = deepcopy(prev['hist'])
        for step in currentHist:
            if step in mergedHist:
                mergedHist[step] = mergedHist[step] + currentHist[step]
            else:
                mergedHist[step] = currentHist[step]
        merged['hist'] = mergedHist
        return merged

    def logStat(self, statValue, epochNr):
        histValues = statValue['hist']
        histKeys = sorted(list(histValues.keys()))
        histHeights = [histValues[k] for k in histKeys]
        plt.figure()
        plt.bar(histKeys, histHeights, width=self.stepSize)
        plt.savefig(os.path.join(self.histDir, self.getStatName() + "_" + str(epochNr) + ".png"))
        return {
            'mean': sum([a*b for a,b in zip (histKeys, histHeights)]) / sum(histHeights)
        }

    def getStatName(self):
        return "reprDiff_" + self.reprType + "_" + self.metric + "_by" + str(self.stepSize)