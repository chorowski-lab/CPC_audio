

import torch.nn as nn
import torch
import numpy as np
import math
from collections import deque

from cpc.model import seDistancesToCentroids

class CentroidModule(nn.Module):

    def __init__(self, settings):  #numCentroids=50, reprDim=256, initAfterEpoch=1):  # initAfterEpoch can be set to -1 for no repr init, just regular
        super().__init__()
        #self.protos = nn.Parameter(torch.randn((self.numProtos, self.reprDim), requires_grad=True) / (5. * math.sqrt(self.reprDim)), requires_grad=True)
        self.dsLen = 0
        self.dsCountEpoch = None
        self.nextExampleNum = 0
        self.seenNr = 0
        self.chosenExamples = None
        self.kmeansInitBatches = None  # will be set below if value given
        self.kmeansReinitEachN = None  # same as above
        self.centerNorm = None # same
        self.batchUpdateQ = None  # same
        self.protoCounts = None
        self.protoSums = None
        #self.inBatch = 0
        self.chosenBatchInputs = []
        self.chosenKMeansBatches = []
        self.numCentroids = settings["numCentroids"]
        self.reprDim = settings["reprDim"]
        self.numPhones = settings["numPhones"]
        #print(f"-------->*************** centermodel numPhones: {self.numPhones}")
        self.mode = settings["mode"]
        if self.mode == "reprInit":
            self.initAfterEpoch = settings["initAfterEpoch"]
            print(f"INIT AFTER EPOCH: {self.initAfterEpoch}")
        if self.mode in ("reprInit", "beginInit"):
            # needed also in reprInit case so that params can be added to optimizer
            # [!!] this is computed here, but only learnt in cpcModel forward!
            self.protos = nn.Parameter(torch.randn((self.numCentroids, self.reprDim), requires_grad=True).cuda() / (5. * math.sqrt(self.reprDim)), requires_grad=True)
        if self.mode == "onlineKmeans":  # someone figured that out and called "sequential k-means" for some reason
                                         # so that's how it's called in literature
            self.initAfterEpoch = settings["initAfterEpoch"]
            self.firstInitNoIters = settings["firstInitNoIters"]
            self.kmeansInitIters = settings["kmeansInitIters"]
            self.kmeansInitBatches = settings["kmeansInitBatches"]
            if self.kmeansInitIters or self.kmeansInitBatches:
                assert self.kmeansInitIters and self.kmeansInitBatches
            self.kmeansReinitEachN = settings["kmeansReinitEachN"]
            self.kmeansReinitUpTo = settings["kmeansReinitUpTo"]
            print(f"INIT AFTER EPOCH: {self.initAfterEpoch}")
            self.keepBatches = settings["onlineKmeansBatches"]
            self.keepBatchesLongTerm = settings["onlineKmeansBatchesLongTerm"]
            self.keepBatchesLongTermWeight = settings["onlineKmeansBatchesLongTermWeight"]
            self.centerNorm = settings["centerNorm"]
            self.batchRecompute = settings["batchRecompute"]
            if self.batchRecompute:
                self.batchUpdateQ = deque()
            self.protos = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
            # TODO to resume, last batches would need to be saved in state - will they be automatically?
            self.currentGlobalBatch = 0
            self.protoCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
            self.protoSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
            self.lastKmBatches = {}  # format:     batch_num: (sums_of_points_assigned_to_each_center, num_points_assigned_to_each_center)
            self.longTermKmBatches = {}
            # TODO think about adding re-initialization (done periodically on last N-batch data with regular k-means or so)

    # before encodings
    def inputsBatchUpdate(self, batch, epochNrs, cpcModel):
        #print(f"--> INPUT BATCH UPDATE ep. {epoch}, {batch.shape}")
        
        with torch.no_grad():
            self.last_input_batch = batch.clone().detach()
            # can be several shorter batches, one per speaker or so, but it is like
            # inputsBatchUpdate, encodingsBatchUpdate are in turns, always
        

    def encodingsBatchUpdate(self, batch, epochNrs, cpcModel, label=None):
        epoch, totalEpochs = epochNrs
        batch = batch.clone().detach()
        if self.dsCountEpoch is None or self.dsCountEpoch == epoch:
            #if self.inBatch == 0:
            #    self.inBatch = (batch.shape[0], batch.shape[1], batch.shape[0] * batch.shape[1])  
                # can be several smaller batches each epoch, one per speaker
            #print(":::::", batch.shape, epochNrs)
            self.dsCountEpoch = epoch
            if self.dsLen == 0:
                print("--> COUNTING DS LEN")
            self.dsLen += batch.shape[0] * batch.shape[1]

        if self.mode in ("reprInit", "onlineKmeans") and epoch == self.initAfterEpoch:  # this epoch has to be > 0
            if self.chosenExamples is None:
                if self.kmeansInitBatches:
                    randNr = max(self.kmeansInitBatches, self.numCentroids)
                else:
                    randNr = self.numCentroids
                self.chosenExamples = np.random.choice(self.dsLen, randNr)
                if self.kmeansInitBatches:
                    self.chosenKMeansCandidateNrs = set(self.chosenExamples[:self.kmeansInitBatches])
                    print(f"--> CHOOSING {self.kmeansInitBatches} BATCHES FOR K-MEANS INIT POINTS;"
                           " will make k-means init with EXAMPLES as starting centers")
                else:
                    self.chosenKMeansCandidateNrs = None
                self.chosenCentroidsCandidateNrs = set(self.chosenExamples[:self.numCentroids])    
                self.chosenExamples = sorted(list(self.chosenExamples))
                self.chosenExamples.append(1000000000000000000000000)  # for convenience and less cases below
                print(f"--> CHOOSING {self.numCentroids} EXAMPLES FOR CENTROID INIT, DSLEN {self.dsLen}: {self.chosenExamples}")
            numHere = batch.shape[0] * batch.shape[1]  #self.inBatch  # hm, several batches can be smaller  #batch.shape[0] * batch.shape[1]
            candidateNr = self.chosenExamples[self.nextExampleNum]
            addedThisBatch = False
            while candidateNr < self.seenNr + numHere:
                offset = candidateNr - self.seenNr
                lineNr = offset // batch.shape[1]
                lineOffset = offset % batch.shape[1]
                #print("!!!", self.protos[self.nextExampleNum].shape, batch[lineNr,lineOffset].detach().shape)
                with torch.no_grad():
                    #self.protos[self.nextExampleNum] = batch[lineNr,lineOffset].detach()
                    if candidateNr in self.chosenCentroidsCandidateNrs:
                        self.chosenBatchInputs.append((self.last_input_batch.clone().cpu(), lineNr, lineOffset))
                        print(f"--> ADDED BATCH EXAMPLE #{self.nextExampleNum}")
                    if self.chosenKMeansCandidateNrs and candidateNr in self.chosenKMeansCandidateNrs and not addedThisBatch:  # both this and above can happen
                        self.chosenKMeansBatches.append(self.last_input_batch.clone().cpu())
                        addedThisBatch = True
                    # TODO ? change this so that batch[...].detach is stored and protos are recomputed after the epoch updates (?)
                    #   ^ could at least check it works
                    # TODO ? or, tbh, maybe should only choose near the end? but actually, earlier protos are also chosen many times sometimes
                    #print(f"--> EXAMPLE #{self.nextExampleNum}: sqlen {(self.protos[self.nextExampleNum]*self.protos[self.nextExampleNum]).sum(-1)}")  #"; {self.protos[self.nextExampleNum]}")
                self.nextExampleNum += 1
                candidateNr = self.chosenExamples[self.nextExampleNum]
            self.seenNr += numHere  #batch.shape[0] * batch.shape[1]

        if self.mode == "onlineKmeans" and epoch > self.initAfterEpoch:

            #print(f"BATCH UPDATE, {self.lastKmBatches.keys()}, {self.longTermKmBatches.keys()}")

            if self.centerNorm:
                batch = self.normLen(batch) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)

            distsSq = seDistancesToCentroids(batch, self.protos)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)

            # add new batch data
            batchSums, closestCounts, labelCounts = self.getBatchSums(batch, closest, label=label)
            self.protoSums += batchSums
            self.protoCounts += closestCounts
            batchToRemember = self.last_input_batch.clone().cpu() if self.batchRecompute else None
            self.lastKmBatches[self.currentGlobalBatch] = (batchSums.cpu(), closestCounts.cpu(), batchToRemember)  # on batchToRemember .cpu() above if not None
            if self.batchRecompute:
                self.batchUpdateQ.append(self.currentGlobalBatch)
            if self.keepBatchesLongTerm:
                weightedSums = self.keepBatchesLongTermWeight*batchSums
                weightedCounts = self.keepBatchesLongTermWeight*closestCounts
                self.protoSums += weightedSums
                self.protoCounts += weightedCounts
                self.longTermKmBatches[self.currentGlobalBatch] = (weightedSums.cpu(), weightedCounts.cpu())

            # subtract old out-of-the-window batch data
            oldBatch = self.currentGlobalBatch - self.keepBatches
            if oldBatch in self.lastKmBatches:
                oldBatchSums, oldBatchCounts, _ = self.lastKmBatches[oldBatch]
                oldBatchSums = oldBatchSums.cuda()
                oldBatchCounts = oldBatchCounts.cuda()
                self.protoSums -= oldBatchSums
                self.protoCounts -= oldBatchCounts
                del self.lastKmBatches[oldBatch]
            if self.keepBatchesLongTerm:
                oldBatch = self.currentGlobalBatch - self.keepBatchesLongTerm
                if oldBatch in self.longTermKmBatches:
                    oldBatchSums, oldBatchCounts = self.longTermKmBatches[oldBatch]
                    oldBatchSums = oldBatchSums.cuda()
                    oldBatchCounts = oldBatchCounts.cuda()
                    self.protoSums -= oldBatchSums
                    self.protoCounts -= oldBatchCounts
                    del self.longTermKmBatches[oldBatch]

            if self.batchRecompute:
                self.updateBatches(epochNrs, cpcModel)

            # re-average centroids
            with torch.no_grad():  # just in case it tries to compute grad
                if self.currentGlobalBatch >= self.keepBatches:
                    self.protos = self.protoSums / torch.clamp(self.protoCounts.view(-1,1), min=1)
                if self.centerNorm:
                    with torch.no_grad():
                        self.protos = self.normLen(self.protos)

            self.currentGlobalBatch += 1

            return {"labelCounts": labelCounts} if self.currentGlobalBatch >= self.keepBatches else None

    def normLen(self, tens):
        # normalization, but not if very very short - to prevent problems during training
        tensLens = torch.sqrt(torch.clamp((tens*tens).sum(-1), min=0))
        return tens / torch.clamp(tensLens.view(*(tensLens.shape), 1), min=1)

    def updateBatches(self, epochNrs, cpcModel):
        updated = 0
        while len(self.batchUpdateQ) > 0 and updated < self.batchRecompute:
            batchNr = self.batchUpdateQ.popleft()
            if batchNr not in self.lastKmBatches:
                continue  # old batch out of window, no update
            oldBatchSums, oldBatchCounts, batch = self.lastKmBatches[batchNr]
            oldBatchSums = oldBatchSums.cuda()
            oldBatchCounts = oldBatchCounts.cuda()
            batch = batch.cuda()
            with torch.no_grad():
                encoded_data = cpcModel(batch, None, None, None, None, epochNrs, False, True)
            if self.centerNorm:
                encoded_data = self.normLen(encoded_data) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            distsSq = seDistancesToCentroids(encoded_data, self.protos)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)
            batchSums, closestCounts, _ = self.getBatchSums(encoded_data, closest)
            self.protoSums -= oldBatchSums
            self.protoCounts -= oldBatchCounts
            self.protoSums += batchSums
            self.protoCounts += closestCounts
            self.lastKmBatches[batchNr] = (batchSums.cpu(), closestCounts.cpu(), batch.cpu())
            self.batchUpdateQ.append(batchNr)
            if self.centerNorm:
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            #print("UPDATED:", batchNr)
            updated += 1

            
            
    def getBatchSums(self, batch, closest, label=None):
        # batch B x n x dim
        # closest B x n
        ###batchExtended = torch.zeros(batch.shape[0], batch.shape[1], self.protos.shape[0], batch.shape[2], dtype=torch.float32).cuda()
        ###firstDim = torch.arange(batch.shape[0]).repeat_interleave(batch.shape[1]).view(batch.shape[0], batch.shape[1])
        #print(firstDim)
        ###secondDim = torch.arange(batch.shape[1]).repeat(batch.shape[0]).view(batch.shape[0], batch.shape[1])
        #print(batchExtended.dtype, batch.dtype)
        ###batchExtended[firstDim, secondDim, closest, :] = batch
        ###batchSums = batchExtended.sum(dim=(0,1))  #[closest] += batch  # only takes last value for index, pathetic
        batchSums = torch.zeros(self.protos.shape[0], batch.shape[2], dtype=torch.float32).cuda()
        for i in range(self.protos.shape[0]):
            batchSums[i] += batch[closest==i, :].sum(dim=(0))
        indices, indicesCounts = torch.unique(closest, return_counts=True)
        closestCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
        closestCounts[indices] += indicesCounts
        if label is not None and self.numPhones:
            label = label.cuda()
            ###labelsAssignment = torch.zeros(batch.shape[0], batch.shape[1], self.protos.shape[0], self.numPhones, dtype=torch.float32).cuda()
            labelsSums = torch.zeros(self.protos.shape[0], self.numPhones, dtype=torch.float32).cuda()
            ###labelsAssignment[firstDim, secondDim, closest, label[firstDim,secondDim]] += 1
            for i in range(self.protos.shape[0]):
                labelclosest = label[closest==i]
                lindices, lindicesCounts = torch.unique(labelclosest, return_counts=True)
                lclosestCounts = torch.zeros(self.numPhones, dtype=torch.float32).cuda()
                lclosestCounts[lindices] += lindicesCounts
                labelsSums[i] += lclosestCounts
            ###labelsSums = labelsAssignment.sum(dim=(0,1))
            return batchSums, closestCounts, labelsSums
        
        return batchSums, closestCounts, None

        

    def printLens(self):
        with torch.no_grad():
            print((self.protos*self.protos).sum(dim=-1))
            
    def getDM(self, epoch):
        protosHere = self.centersForStuff(epoch)
        if protosHere is None:
            return None
        DMsq = seDistancesToCentroids(protosHere, protosHere).view(protosHere.shape[0], protosHere.shape[0])
        return torch.sqrt(torch.clamp(DMsq, min=0))

    def epochUpdate(self, epochNrs, cpcModel):  # after that epoch
        epoch, allEpochs = epochNrs
        if self.mode in ("reprInit", "onlineKmeans"):
            if epoch == self.initAfterEpoch or \
                (epoch > self.initAfterEpoch and self.kmeansReinitEachN and (epoch - self.initAfterEpoch) % self.kmeansReinitEachN == 0 and \
                (not self.kmeansReinitUpTo or epoch < self.kmeansReinitUpTo)):   

                with torch.no_grad():

                    self.currentGlobalBatch = 0  # to prevent pushing with incomplete means
                    # to remove info that will be invalid with new clusters
                    self.lastKmBatches = {}
                    self.longTermKmBatches = {}
                    self.protoCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
                    self.protoSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
                    if self.batchUpdateQ is not None:
                        self.batchUpdateQ.clear()

                    print("K-MEANS CENTERS REINIT FROM REPRESENTATIONS")
                    self.initKmeansCenters(epochNrs, cpcModel)  # initialize centroids
                    if self.kmeansInitBatches and (not self.firstInitNoIters or epoch != self.initAfterEpoch):
                        print("K-MEANS CENTERS REINIT K-MEANS IMPROVE")
                        for i in range(self.kmeansInitIters):  # perform k-means with initizalized centroids
                            print("new kmeans epoch")
                            self.kmeansEpoch(epochNrs, cpcModel)  # performs one epoch, moving the centroids

                # for i, (batchData, lineNr, lineOffset) in enumerate(self.chosenBatchInputs):
                #     with torch.no_grad():
                #         encoded_data = cpcModel(batchData, None, None, None, None, epochNrs, False, True)  # c_feature, encoded_data, label, pushLoss
                #         self.protos[i] = encoded_data[lineNr,lineOffset]
                #         print(f"--> EXAMPLE #{i}: sqlen {(self.protos[i]*self.protos[i]).sum(-1)}")  #"; {self.protos[i]}")


    def initKmeansCenters(self, epochNrs, cpcModel):
        for i, (batchData, lineNr, lineOffset) in enumerate(self.chosenBatchInputs):
            with torch.no_grad():
                batchData = batchData.cuda()
                encoded_data = cpcModel(batchData, None, None, None, None, epochNrs, False, True)  # c_feature, encoded_data, label, pushLoss
                self.protos[i] = encoded_data[lineNr,lineOffset]
                # [!!!] here it's not normed, it's normed before any distance operation or before giving it outside
                print(f"--> EXAMPLE #{i}: sqlen {(self.protos[i]*self.protos[i]).sum(-1)}")  #"; {self.protos[i]}")


    def kmeansEpoch(self, epochNrs, cpcModel):
        # this assumes centroids are already initialized
        newCentersSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
        newCentersCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
        print(f"ACTUAL BATCHES for k-means init epoch: {len(self.chosenKMeansBatches)}")
        for i, batch in enumerate(self.chosenKMeansBatches):
            batch = batch.cuda()
            encoded_data = cpcModel(batch, None, None, None, None, epochNrs, False, True)
            if self.centerNorm:
                encoded_data = self.normLen(encoded_data) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            distsSq = seDistancesToCentroids(encoded_data, self.protos)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)
            # add new batch data
            batchSums, closestCounts, _ = self.getBatchSums(encoded_data, closest)
            newCentersSums += batchSums
            newCentersCounts += closestCounts
        with torch.no_grad():  # just in case it tries to compute grad
            self.protos = newCentersSums / torch.clamp(newCentersCounts.view(-1,1), min=1)


    def centersForStuff(self, epoch):
        # TODO can add option to always return, for some MOD cases etc
        if self.centerNorm:
            with torch.no_grad():
                self.protos = self.normLen(self.protos)
        if self.mode == "reprInit":
            if epoch <= self.initAfterEpoch:
                return None
            else:
                return self.protos
        if self.mode == "beginInit":
            return self.protos

        if self.mode == "onlineKmeans":
            # count starts after init, but more fireproof with epoch check
            #print("!!!", self.initAfterEpoch, self.currentGlobalBatch)
            if epoch > self.initAfterEpoch and self.currentGlobalBatch >= self.keepBatches:
                return self.protos
            else:
                return None
        






if __name__ == "__main__":

    # online kmeans test

    distsSq = torch.tensor([[[1,2], [4,3], [5,6]], [[1,2], [4,3], [5,6]]], dtype=float)
    batch = torch.tensor([[[7,7], [2,2], [3,3]], [[7,7], [2,2], [3,3]]], dtype=float)
    protos = torch.tensor([[1,7], [123,2]], dtype=float)
    closest = distsSq.argmin(-1)

    cm = CentroidModule({
        "mode": "onlineKmeans",
        "onlineKmeansBatches": 2, 
        "reprDim": 2,
        "numCentroids": 4,
        "initAfterEpoch": 1})

    print(cm.getBatchSums(batch, closest))


    batch1 = torch.tensor([[[17,17], [2,2], [31,31]], [[17,17], [2,2], [31,31]]], dtype=float)
    batch2 = torch.tensor([[[18,18], [2,2], [32,32]], [[18,18], [2,2], [32,32]]], dtype=float)
    batch3 = torch.tensor([[[19,19], [2,2], [33,33]], [[19,19], [2,2], [33,33]]], dtype=float)

    cpcModelFake = lambda batch, a, b, c, d: (None, batch, None, None)

    for ep in range(4):
        for batch in (batch1, batch2, batch3):
            cm.inputsBatchUpdate(batch, ep)
            cm.encodingsBatchUpdate(batch, ep)
            print("\n! ", cm.centersForStuff(ep))
        cm.epochUpdate((ep, 4), cpcModelFake)
        print("\n!!! ", cm.centersForStuff(ep))












