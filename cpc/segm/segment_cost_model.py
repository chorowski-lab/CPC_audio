

from math import sqrt

class SegmentCostModel:


    def __init__(self, segmSettings):
        self.batchesMem = segmSettings["batchesMem"]
        self.mem = {}   # seems Torch can just save anything and not only models and their params
                        # so will just be saving and loading class instance, no need to store everything as pseudo-parameters
                        # of a pseudo-torch-model, can normally save inside checkpoint without that
        self.globBatchNr = 0
        self.cnt = 0.
        self.memSum = 0.
        self.memSqSum = 0.
        self.memSumK = 0.
        self.memSqSumK = 0.
        self.multMem = 1  # in multi-GPU case; could actually do stuff without that, but did it similarly as k-means mem; TODO change??

    # this should be ONLY invoked in main training, as it affects the estimate; TODO maybe make some other stat-only method??
    def batchUpdate(self, batchDataCostSegmForKTens, batchDataActualKTens):
        # batchData: 1 x nGPU tensor with max segm costs for wanted num segms in batch (to be averaged)
        #newCostAvg = batchDataCostSegmForKTens.mean().item()
        #newKAvg = batchDataActualKTens.mean().item()
        assert batchDataCostSegmForKTens.shape[0] == batchDataActualKTens.shape[0]
        if batchDataCostSegmForKTens.shape[0] != 0:
            self.multMem = batchDataCostSegmForKTens.shape[0]
        for i in range(batchDataCostSegmForKTens.shape[0]):
            newCost = batchDataCostSegmForKTens[i][0].item()
            newK = batchDataActualKTens[i][0].item()
            # [!] K stats could be disrupted by smaller batches, but those are NOT invoked to update
            #%#print("!!!!!!!!! costEntry:", newCost, newK)
            self.mem[self.globBatchNr] = (newCost, newK)
            self.cnt += 1
            self.memSum += newCost
            self.memSqSum += newCost*newCost
            self.memSumK += newK
            self.memSqSumK += newK*newK
            self.globBatchNr += 1
            oldBatch = self.globBatchNr - self.multMem*self.batchesMem
            if oldBatch in self.mem:
                oldCost, oldK = self.mem[oldBatch]
                self.cnt -= 1
                self.memSum -= oldCost
                self.memSqSum -= oldCost*oldCost
                self.memSumK -= oldK
                self.memSqSumK -= oldK*oldK
                del self.mem[oldBatch]

    def showCurrentStats(self):
        kMean = self.memSumK / self.cnt
        kVar = max((self.memSqSumK / self.cnt) - (kMean)**2, 0.)
        costMean = self.memSum / self.cnt
        costVar = max((self.memSqSum / self.cnt) - (costMean)**2, 0.)
        print("-----------")
        print(f"cost of merging segments across last {self.batchesMem}*{self.multMem} bacthes: mean {costMean}, stdev {sqrt(costVar)}, var {costVar}")
        print(f"number of left segments across last {self.batchesMem}*{self.multMem} bacthes: mean {kMean}, stdev {sqrt(kVar)}, var {kVar}")
        print("-----------")


    def getCurrentMaxCostEstimator(self):
        if self.globBatchNr < self.multMem*self.batchesMem:
            return None
        return self.memSum / self.cnt  # well, here cnt has to be self.batchesMem actually

    

