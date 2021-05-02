

import torch.nn as nn
import torch
import numpy as np
import math

from cpc.model import seDistancesToCentroidsCpy

class CentroidModule(nn.Module):

    def __init__(self, settings):  #numCentroids=50, reprDim=256, initAfterEpoch=1):  # initAfterEpoch can be set to -1 for no repr init, just regular
        super().__init__()
        #self.protos = nn.Parameter(torch.randn((self.numProtos, self.reprDim), requires_grad=True) / (5. * math.sqrt(self.reprDim)), requires_grad=True)
        self.dsLen = 0
        self.nextExampleNum = 0
        self.seenNr = 0
        self.chosenExamples = None
        #self.inBatch = 0
        self.chosenBatchInputs = []
        self.numCentroids = settings["numCentroids"]
        self.reprDim = settings["reprDim"]
        self.initAfterEpoch = settings["initAfterEpoch"]
        print(f"INIT AFTER EPOCH: {self.initAfterEpoch}")
        # needed so that params can be added to optimizer
        self.protos = nn.Parameter(torch.randn((self.numCentroids, self.reprDim), requires_grad=True) / (5. * math.sqrt(self.reprDim)), requires_grad=True)

    # before encodings
    def inputsBatchUpdate(self, batch, epoch):
        #print(f"--> INPUT BATCH UPDATE ep. {epoch}, {batch.shape}")
        
        with torch.no_grad():
            self.last_input_batch = batch.clone().detach()
            # can be several shorter batches, one per speaker or so, but it is like
            # inputsBatchUpdate, encodingsBatchUpdate are in turns, always
        

    def encodingsBatchUpdate(self, batch, epoch):
        if epoch == self.initAfterEpoch - 1:
            #if self.inBatch == 0:
            #    self.inBatch = (batch.shape[0], batch.shape[1], batch.shape[0] * batch.shape[1])  
                # can be several smaller batches each epoch, one per speaker
            #print(batch.shape)
            if self.dsLen == 0:
                print("--> COUNTING DS LEN")
            self.dsLen += batch.shape[0] * batch.shape[1]

        if epoch == self.initAfterEpoch:  # this epoch has to be > 0
            if self.chosenExamples is None:
                self.chosenExamples = np.random.choice(self.dsLen, self.numCentroids)
                self.chosenExamples = sorted(list(self.chosenExamples))
                self.chosenExamples.append(1000000000000000000000000)  # for convenience and less cases below
                print(f"--> CHOOSING {self.numCentroids} EXAMPLES FOR INIT, DSLEN {self.dsLen}: {self.chosenExamples}")
            numHere = batch.shape[0] * batch.shape[1]  #self.inBatch  # hm, several batches can be smaller  #batch.shape[0] * batch.shape[1]
            candidateNr = self.chosenExamples[self.nextExampleNum]
            while candidateNr < self.seenNr + numHere:
                offset = candidateNr - self.seenNr
                lineNr = offset // batch.shape[1]
                lineOffset = offset % batch.shape[1]
                #print("!!!", self.protos[self.nextExampleNum].shape, batch[lineNr,lineOffset].detach().shape)
                with torch.no_grad():
                    #self.protos[self.nextExampleNum] = batch[lineNr,lineOffset].detach()
                    self.chosenBatchInputs.append((self.last_input_batch.clone(), lineNr, lineOffset))
                    print(f"--> ADDED BATCH EXAMPLE #{self.nextExampleNum}")
                    # TODO ? change this so that batch[...].detach is stored and protos are recomputed after the epoch updates (?)
                    #   ^ could at least check it works
                    # TODO ? or, tbh, maybe should only choose near the end? but actually, earlier protos are also chosen many times sometimes
                    #print(f"--> EXAMPLE #{self.nextExampleNum}: sqlen {(self.protos[self.nextExampleNum]*self.protos[self.nextExampleNum]).sum(-1)}")  #"; {self.protos[self.nextExampleNum]}")
                self.nextExampleNum += 1
                candidateNr = self.chosenExamples[self.nextExampleNum]
            self.seenNr += numHere  #batch.shape[0] * batch.shape[1]
            

    def printLens(self):
        with torch.no_grad():
            print((self.protos*self.protos).sum(dim=-1))
            
        
    def epochUpdate(self, epochNrs, cpcModel):  # after that epoch
        epoch, allEpochs = epochNrs
        if epoch == self.initAfterEpoch:
            for i, (batchData, lineNr, lineOffset) in enumerate(self.chosenBatchInputs):
                with torch.no_grad():
                    c_feature, encoded_data, label, pushLoss = cpcModel(batchData, None, None, epochNrs, False)
                    self.protos[i] = encoded_data[lineNr,lineOffset]
                    print(f"--> EXAMPLE #{i}: sqlen {(self.protos[i]*self.protos[i]).sum(-1)}")

        

    def centersForStuff(self, epoch):
        # TODO can add option to always return, for some FCM cases etc
        if epoch <= self.initAfterEpoch:
            return None
        else:
            return self.protos
        



















