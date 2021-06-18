# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from .seq_alignment import collapseLabelChain
from .custom_layers import EqualizedLinear, EqualizedConv1d


class FFNetwork(nn.Module):
    def __init__(self, din, dout, dff, dropout):
        super(FFNetwork, self).__init__()
        self.lin1 = EqualizedLinear(din, dff, bias=True, equalized=True)
        self.lin2 = EqualizedLinear(dff, dout, bias=True, equalized=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.drop(self.relu(self.lin1(x))))


class ShiftedConv(nn.Module):
    def __init__(self, dimOutputAR, dimOutputEncoder, kernelSize):
        super(ShiftedConv, self).__init__()
        self.module = EqualizedConv1d(dimOutputAR, dimOutputEncoder,
                                      kernelSize, equalized=True,
                                      padding=0)
        self.kernelSize = kernelSize

    def forward(self, x):

        # Input format: N, S, C -> need to move to N, C, S
        N, S, C = x.size()
        x = x.permute(0, 2, 1)

        padding = torch.zeros(N, C, self.kernelSize - 1, device=x.device)
        x = torch.cat([padding, x], dim=2)
        x = self.module(x)
        x = x.permute(0, 2, 1)
        return x


class PredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(PredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR

        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'ffd':
                self.predictors.append(
                    FFNetwork(dimOutputAR, dimOutputEncoder,
                              dimOutputEncoder, 0))
            elif rnnMode == 'conv4':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            elif rnnMode == 'conv8':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            elif rnnMode == 'conv12':
                self.predictors.append(
                    ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from cpc.transformers import buildTransformerAR
                self.predictors.append(
                    buildTransformerAR(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False))
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates):

        assert(len(candidates) == len(self.predictors))
        out = []

        # UGLY
        if isinstance(self.predictors[0], EqualizedConv1d):
            c = c.permute(0, 2, 1)

        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            if isinstance(self.predictors[k], EqualizedConv1d):
                locC = locC.permute(0, 2, 1)
            if self.dropout is not None:
                locC = self.dropout(locC)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
        return out

    
class TimeAlignedPredictionNetwork(nn.Module):

    def __init__(self,
                 nPredicts,
                 dimOutputAR,
                 dimOutputEncoder,
                 rnnMode=None,
                 dropout=False,
                 sizeInputSeq=116):

        super(TimeAlignedPredictionNetwork, self).__init__()
        self.predictors = nn.ModuleList()
        self.RESIDUAL_STD = 0.01
        self.dimOutputAR = dimOutputAR
        print("LOADING TIME ALIGNED PRED simple")
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        for i in range(nPredicts):
            if rnnMode == 'RNN':
                self.predictors.append(
                    nn.RNN(dimOutputAR, dimOutputEncoder))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'LSTM':
                self.predictors.append(
                    nn.LSTM(dimOutputAR, dimOutputEncoder, batch_first=True))
                self.predictors[-1].flatten_parameters()
            elif rnnMode == 'ffd':
                self.predictors.append(
                    FFNetwork(dimOutputAR, dimOutputEncoder,
                              dimOutputEncoder, 0))
            # elif rnnMode == 'conv4':
            #     self.predictors.append(
            #         ShiftedConv(dimOutputAR, dimOutputEncoder, 4))
            # elif rnnMode == 'conv8':
            #     self.predictors.append(
            #         ShiftedConv(dimOutputAR, dimOutputEncoder, 8))
            # elif rnnMode == 'conv12':
            #     self.predictors.append(
            #         ShiftedConv(dimOutputAR, dimOutputEncoder, 12))
            elif rnnMode == 'transformer':
                from cpc.transformers import buildTransformerAR
                self.predictors.append(
                    buildTransformerAR(dimOutputEncoder,
                                       1,
                                       sizeInputSeq,
                                       False))
            else:
                self.predictors.append(
                    nn.Linear(dimOutputAR, dimOutputEncoder, bias=False))
                if dimOutputEncoder > dimOutputAR:
                    residual = dimOutputEncoder - dimOutputAR
                    self.predictors[-1].weight.data.copy_(torch.cat([torch.randn(
                        dimOutputAR, dimOutputAR), self.RESIDUAL_STD * torch.randn(residual, dimOutputAR)], dim=0))

    def forward(self, c, candidates, predictedLengths):

        assert(len(candidates) == len(self.predictors))
        out = []

        predictedLengths = torch.sigmoid(predictedLengths)

        # predictor choice for each frame - calculate tensor which will tell what predictor and use it
        # to parallelize later on indices == k, without changing shapes and restoring what was where
        # but need to modify len passed for each future-frame prediction :/

        # calc prefsums from each place
        #predictedLengths = torch.clamp(predictedLengths, max=1.)  # TODO think if stuff will propagate if done like that; maybe should normalize to max length? or not...
        #^#print("predLen", predictedLengths)
        # TODO ^ another option - just do nothing with those, if too long stuff will jump to the last predictor
        # TODO somehow initialize weights output to be 0.5 or so or sth!
        # TODO ^ well, maybe actually just normalize those lengths somewhat, e.g. in the whole batch?
        #        BUT SIMILAR PROBLEM AS WITH HIERAR, WOULD LIKE NOT PER BATCH BUT SOME SUM AVG IN LAST BATCHES
        # ---> WELL, MAYBE ADD BIG LOSS TELLING THAT THIS MUST BE <= THAN 1 , AND CLAMP
        #      also, can teach all predictors but those further with much smaller weight? BUT THIS WOULD COST PERHAPS? WELL, RATHER ONLY A BIT
        predictedLengthsSum = predictedLengths.cumsum(dim=1)
        #^#print("predLenSum", predictedLengthsSum)
        moreLengths = predictedLengthsSum.view(1,predictedLengthsSum.shape[0],predictedLengthsSum.shape[1]).cuda().repeat(len(candidates),1,1)
        #^#print(moreLengths.shape)
        for i in range(1,len(candidates)+1):
            #^#print(moreLengths[i-1].shape)
            #^#print("*", i, moreLengths[i-1], torch.roll(moreLengths[i-1], shifts=(0,-i), dims=(0,1)))
            moreLengths[i-1] = torch.roll(moreLengths[i-1], shifts=(0,-i), dims=(0,1)) - predictedLengthsSum
        moreLengths = moreLengths[:,:,:c.shape[1]]  # cut rubbish at the end which is not being predicted
        #^#print("moreLen", moreLengths)

        # for each nr of frames in future separately,
        # calc and switch last elements in c as lengths, and also get predictor choices
        toPredCenters = (torch.arange(len(candidates)).cuda()).view(1,1,1,-1)
        lengthsDists = torch.abs(moreLengths.view(moreLengths.shape[0],moreLengths.shape[1],moreLengths.shape[2],1) - toPredCenters)
        #weights, closest = torch.topk(lengthsDists, 2, dim=-1, largest=False)
        #weights = 1 - weights
        #w1 = weights[0]
        #w2 = weights[1]
        #weights[0,w2<0] = 1  # in places not between two predictors (<0.5 on borders), assign all weight to closest one
        #weights = torch.clamp(weights, min=0)
        #^#print("lengthsDists", lengthsDists)
        weights = torch.exp(-2.*lengthsDists)
        weightsNorms = weights.sum(-1)
        #^#print("weightsUnnormed", weights)
        #^#print("weightNorms", weightsNorms)
        weights = weights / weightsNorms.view(*(weightsNorms.shape),1)
        #^#print("weights", weights)

        #^#print("shapes:", c.shape, predictedLengthsSum.shape, predictedLengths.shape)
        #c = c.view(1,*(c.shape)).repeat(len(candidates),1,1,1)
        c = c.clone()  # because of not-inplace view things
        ###c[:,:,-2] = predictedLengthsSum[:,:-len(candidates)]  #  [:,:,:,-1]  moreLengths
        # ^ this seems like a very bad idea after some rethinking - teaches all previous lengths in the batch from local predictions (idea was for it to make diffs, but well, it can do sth else)
        # frame lengths are now at -2, they are given as part of input c, but can also put there again to be sure
        c[:,:,-1] = predictedLengths[:,:-len(candidates)].detach()  #.requires_grad_()
        c[:,:,-2] = predictedLengthsSum[:,:-len(candidates)].detach()
        #^#print("c:", c)

        
        # UGLY   ; not sure if will work
        # if isinstance(self.predictors[0], EqualizedConv1d):
        #     c = c.permute(0, 2, 1)


        # there's a problem with how predictors look like - those utilize frame-constant thing heavily
        # as they are e.g. LSTM/transformers ; simple feedforward net would perhaps be a lot worse
        # so this idea can't really be made like that (?)
        # or could but with a lot of complications
        # [!!! v]
        # it can be done with one bigger predictor though
        # but then there is another problem - this predictor still needs to be run 12 times?
        # or only 1 time but outputting 12 times as big output (and having 12 to-predict durations on input possibly; or just in-place durations?)
        # TBH if transformer is used, it will see those future durations, so maybe it will use them somehow
        # would need to pass both length and how far in the future we need to predict; or lengths and its cumsum?

        # [!!! v]
        # well, this multi-predictor option can actually be done but then only pass durations (and to cumsums)
        # and assume duration diff to be predictor's number for each one
        # would then just need to compute each predictor on whole input and then,
        # do this weighting on wanted things (each prediction could then have several positives) - so in a way actual diff and not round(diff) is also known

        # TODO maybe my model variant should modify mask to see durations in the future??? or rather not - could use this for cheating
        # [!!!] actually, the model shouldn't take after what time to predict as input
        #       as this would be cheating - it would see info from the future and would perhaps try to encode sth else than duration there

        predictsPerPredictor = torch.zeros(1,*(c.shape)).cuda().repeat(len(candidates),1,1,1)  #.view(c.shape[0],c.shape[1],c.shape[2],c.shape[3])  #.repeat(1,1,1,2,1)
        
        #^#print("devices:", c.device, predictsPerPredictor.device, weights.device, predictedLengths.device, predictedLengthsSum.device)

        
        for k in range(len(self.predictors)):

            locC = self.predictors[k](c)
            if isinstance(locC, tuple):
                locC = locC[0]
            predictsPerPredictor[k] = locC

        #^#print("ppp", predictsPerPredictor.shape)

        predsWeighted = predictsPerPredictor.view(predictsPerPredictor.shape[0], predictsPerPredictor.shape[1], predictsPerPredictor.shape[2], 1, predictsPerPredictor.shape[3])
        predsWeighted = predsWeighted * weights.view(weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3], 1)
        
        #^#print("predsWeightedNoSum", predsWeighted.shape)

        # predictions = torch.zeros_like(c).view(c.shape[0],c.shape[1],c.shape[2],1,c.shape[3]).repeat(1,1,1,2,1)
        # for k in range(len(self.predictors)):

        #     # correct time distance weights already swapped inside c
        #     locC1 = self.predictors[k](c[closest[0]==k,:])  #self.predictors[k](c)
        #     if isinstance(locC1, tuple):
        #         locC1 = locC1[0]
        #     predictions[closest[0]==k,0,:] = locC1*weights[0,closest[0]==k]
        #     locC2 = self.predictors[k](c[closest[1]==k,:])
        #     if isinstance(locC2, tuple):
        #         locC2 = locC2[0]
        #     predictions[closest[0]==k,1,:] = locC2*weights[1,closest[1]==k]

        #predictions = predictions.sum(dim=-2)  # sum weighted stuff
        predsWeighted = predsWeighted.sum(dim=-2)
        #^#print("predsWeighted", predsWeighted.shape)
        # now predictions numPred x B x N x Dim

        for k in range(len(candidates)):  #(len(self.predictors)):  # same, but clearer
            # if isinstance(locC, tuple):
            #     locC = locC[0]
            # if isinstance(self.predictors[k], EqualizedConv1d):
            #     locC = locC.permute(0, 2, 1)
            locC = predsWeighted[k]  # B x (N-pred) x Dim
            if self.dropout is not None:
                locC = self.dropout(locC)
            #^#print("locC", locC.shape, len(candidates), candidates[0].shape)
            locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))[:,:,:,:-2]  # cut length and length sum dims
            #^#print("view", locC.shape, candidates[k].shape)
            outK = (locC*candidates[k]).mean(dim=3)
            out.append(outK)
        return out


    # def forward(self, c, candidates, predictedLengths):

    #     assert(len(candidates) == len(self.predictors))
    #     out = []

    #     # predictor choice for each frame - calculate tensor which will tell what predictor and use it
    #     # to parallelize later on indices == k, without changing shapes and restoring what was where
    #     # but need to modify len passed for each future-frame prediction :/

    #     # calc prefsums from each place
    #     predictedLengths = torch.clamp(predictedLengths, max=1.)  # TODO think if stuff will propagate if done like that; maybe should normalize to max length? or not...
    #     # TODO ^ another option - just do nothing with those, if too long stuff will jump to the last predictor
    #     # TODO somehow initialize weights output to be 0.5 or so or sth!
    #     # TODO ^ well, maybe actually just normalize those lengths somewhat, e.g. in the whole batch?
    #     #        BUT SIMILAR PROBLEM AS WITH HIERAR, WOULD LIKE NOT PER BATCH BUT SOME SUM AVG IN LAST BATCHES
    #     predictedLengths = predictedLengths.cumsum(predictedLengths, dim=1)
    #     moreLengths = predictedLengths.repeat(1,1,len(candidates))
    #     for i in range(1,len(candidates)+1):
    #         moreLengths[i-1] = moreLengths[i-1].rotate(0,-i) - predictedLengths
    #     moreLengths = moreLengths[:,:,:c.shape[1]]  # cut rubbish at the end which is not being predicted

    #     # for each nr of frames in future separately,
    #     # calc and switch last elements in c as lengths, and also get predictor choices
    #     toPredCenters = (torch.arange(len(candidates)).cuda() + 0.5).view(1,1,1,-1)
    #     lengthsDists = torch.abs(moreLengths.view(moreLengths.shape[0],moreLengths.shape[1],moreLengths.shape[2],1) - toPredCenters)
    #     weights, closest = torch.topk(lengthsDists, 2, dim=-1, largest=False)
    #     weights = 1 - weights
    #     #w1 = weights[0]
    #     w2 = weights[1]
    #     weights[0,w2<0] = 1  # in places not between two predictors (<0.5 on borders), assign all weight to closest one
    #     weights = torch.clamp(weights, min=0)

    #     c = c.view(1,*(c.shape)).repeat(len(candidates),1,1,1)
    #     c[:,:,:,-1] = moreLengths


    #     # UGLY   ; not sure if will work
    #     # if isinstance(self.predictors[0], EqualizedConv1d):
    #     #     c = c.permute(0, 2, 1)

        
    #     predictions = torch.zeros_like(c).view(c.shape[0],c.shape[1],c.shape[2],1,c.shape[3]).repeat(1,1,1,2,1)
    #     for k in range(len(self.predictors)):

    #         # correct time distance weights already swapped inside c
    #         locC1 = self.predictors[k](c[closest[0]==k,:])  #self.predictors[k](c)
    #         if isinstance(locC1, tuple):
    #             locC1 = locC1[0]
    #         predictions[closest[0]==k,0,:] = locC1*weights[0,closest[0]==k]
    #         locC2 = self.predictors[k](c[closest[1]==k,:])
    #         if isinstance(locC2, tuple):
    #             locC2 = locC2[0]
    #         predictions[closest[0]==k,1,:] = locC2*weights[1,closest[1]==k]

    #     predictions = predictions.sum(dim=-2)  # sum weighted stuff
    #     # now predictions numPred x B x N x Dim

    #     for k in range(len(candidates)):  #(len(self.predictors)):  # same, but clearer
    #         # if isinstance(locC, tuple):
    #         #     locC = locC[0]
    #         # if isinstance(self.predictors[k], EqualizedConv1d):
    #         #     locC = locC.permute(0, 2, 1)
    #         locC = predictions[k]  # B x N x Dim
    #         if self.dropout is not None:
    #             locC = self.dropout(locC)
    #         locC = locC.view(locC.size(0), 1, locC.size(1), locC.size(2))
    #         outK = (locC*candidates[k]).mean(dim=3)
    #         out.append(outK)
    #     return out


class BaseCriterion(nn.Module):

    def warmUp(self):
        return False

    def update(self):
        return


class NoneCriterion(BaseCriterion):
    def __init__(self):
        super(NoneCriterion, self).__init__()

    def forward(self, cFeature, encodedData, label):
        return torch.zeros(1, 1, device=cFeature.device), \
            torch.zeros(1, 1, device=cFeature.device)


class CPCUnsupersivedCriterion(BaseCriterion):

    def __init__(self,
                 nPredicts,             # Number of steps
                 dimOutputAR,           # Dimension of G_ar
                 dimOutputEncoder,      # Dimension of the convolutional net
                 negativeSamplingExt,   # Number of negative samples to draw
                 mode=None,
                 rnnMode=False,
                 dropout=False,
                 speakerEmbedding=0,
                 nSpeakers=0,
                 sizeInputSeq=128,
                 lengthInARsettings=None):

        super(CPCUnsupersivedCriterion, self).__init__()
        if speakerEmbedding > 0:
            print(
                f"Using {speakerEmbedding} speaker embeddings for {nSpeakers} speakers")
            self.speakerEmb = torch.nn.Embedding(nSpeakers, speakerEmbedding)
            dimOutputAR += speakerEmbedding
        else:
            self.speakerEmb = None

        self.modelLengthInARsimple = lengthInARsettings["modelLengthInARsimple"]

        if not self.modelLengthInARsimple:
            self.wPrediction = PredictionNetwork(
                nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
                dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
        else:
            self.wPrediction = TimeAlignedPredictionNetwork(
                nPredicts, dimOutputAR, dimOutputEncoder, rnnMode=rnnMode,
                dropout=dropout, sizeInputSeq=sizeInputSeq - nPredicts)
        self.nPredicts = nPredicts
        self.negativeSamplingExt = negativeSamplingExt
        self.lossCriterion = nn.CrossEntropyLoss()

        if mode not in [None, "reverse"]:
            raise ValueError("Invalid mode")

        self.mode = mode

    def sampleClean(self, encodedData, windowSize):

        batchSize, nNegativeExt, dimEncoded = encodedData.size()
        outputs = []

        negExt = encodedData.contiguous().view(-1, dimEncoded)
        # Draw nNegativeExt * batchSize negative samples anywhere in the batch
        batchIdx = torch.randint(low=0, high=batchSize,
                                 size=(self.negativeSamplingExt
                                       * windowSize * batchSize, ),
                                 device=encodedData.device)

        seqIdx = torch.randint(low=1, high=nNegativeExt,
                               size=(self.negativeSamplingExt
                                     * windowSize * batchSize, ),
                               device=encodedData.device)

        baseIdx = torch.arange(0, windowSize, device=encodedData.device)
        baseIdx = baseIdx.view(1, 1,
                               windowSize).expand(1,
                                                  self.negativeSamplingExt,
                                                  windowSize).expand(batchSize, self.negativeSamplingExt, windowSize)
        seqIdx += baseIdx.contiguous().view(-1)
        seqIdx = torch.remainder(seqIdx, nNegativeExt)

        extIdx = seqIdx + batchIdx * nNegativeExt
        negExt = negExt[extIdx].view(batchSize, self.negativeSamplingExt,
                                     windowSize, dimEncoded)

        labelLoss = torch.zeros((batchSize * windowSize),
                                dtype=torch.long,
                                device=encodedData.device)

        for k in range(1, self.nPredicts + 1):

            # Positive samples
            if k < self.nPredicts:
                posSeq = encodedData[:, k:-(self.nPredicts-k)]
            else:
                posSeq = encodedData[:, k:]

            posSeq = posSeq.view(batchSize, 1, posSeq.size(1), dimEncoded)
            fullSeq = torch.cat((posSeq, negExt), dim=1)
            outputs.append(fullSeq)

        return outputs, labelLoss

    def getInnerLoss(self):

        return "orthoLoss", self.orthoLoss * self.wPrediction.orthoCriterion()

    def forward(self, cFeature, predictedLengths, encodedData, label, captureOptions=None):

        if self.mode == "reverse":
            encodedData = torch.flip(encodedData, [1])
            cFeature = torch.flip(cFeature, [1])

        batchSize, seqSize, dimAR = cFeature.size()
        windowSize = seqSize - self.nPredicts

        if self.modelLengthInARsimple:
            predictedFrameLengths = cFeature[:,:,-1]

        cFeature = cFeature[:, :windowSize]  # ^ need to extract pred lengths for future before that step
        # TODO ^ is a bad idea anyway, this should be cut after predictions (e.g. for transformer)

        sampledData, labelLoss = self.sampleClean(encodedData, windowSize)

        if self.speakerEmb is not None:
            l_ = label.view(batchSize, 1).expand(batchSize, windowSize)
            embeddedSpeaker = self.speakerEmb(l_)
            cFeature = torch.cat([cFeature, embeddedSpeaker], dim=2)

        if self.modelLengthInARsimple:
            predictions = self.wPrediction(cFeature, sampledData, predictedFrameLengths)
        else:
            predictions = self.wPrediction(cFeature, sampledData)

        captureRes = None
        if captureOptions != None:
            for o in captureOptions:
                assert o in ('pred',)
            captureRes = {}
            if 'pred' in captureOptions:
                assert False   # not supported yet, predictions here are in some very weird format it seems
                captureRes['pred'] = None

        outLosses = [0 for x in range(self.nPredicts)]
        outAcc = [0 for x in range(self.nPredicts)]

        for k, locPreds in enumerate(predictions[:self.nPredicts]):
            locPreds = locPreds.permute(0, 2, 1)
            locPreds = locPreds.contiguous().view(-1, locPreds.size(2))
            lossK = self.lossCriterion(locPreds, labelLoss)
            outLosses[k] += lossK.view(1, -1)
            _, predsIndex = locPreds.max(1)
            outAcc[k] += torch.sum(predsIndex == labelLoss).float().view(1, -1)

        return torch.cat(outLosses, dim=1), \
            torch.cat(outAcc, dim=1) / (windowSize * batchSize), \
                captureRes


class SpeakerCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nSpeakers, nLayers=1):

        super(SpeakerCriterion, self).__init__()
        # self.linearSpeakerClassifier = nn.Linear(
        #     dimEncoder, nSpeakers)
        if nLayers == 1:
            self.linearSpeakerClassifier = nn.Linear(dimEncoder, nSpeakers)
        else:
            outLayers = [nn.Linear(dimEncoder, nSpeakers)]
            for l in range(nLayers - 1):
                outLayers.append(nn.ReLU())
                outLayers.append(nn.Linear(nSpeakers, nSpeakers))
            self.linearSpeakerClassifier = nn.Sequential(*outLayers)
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        batchSize = cFeature.size(0)
        cFeature = cFeature[:, -1, :]
        cFeature = cFeature.view(batchSize, -1)
        predictions = self.linearSpeakerClassifier(cFeature)

        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc

class SpeakerDoubleCriterion(BaseCriterion):

    def __init__(self, dimEncoder, dimInter, nSpeakers):

        super(SpeakerDoubleCriterion, self).__init__()
        self.linearSpeakerClassifier = nn.Sequential(nn.Linear(dimEncoder, dimInter), 
            nn.Linear(dimInter, nSpeakers))
        self.lossCriterion = nn.CrossEntropyLoss()
        self.entropyCriterion = nn.LogSoftmax(dim=1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        batchSize = cFeature.size(0)
        cFeature = cFeature[:, -1, :]
        cFeature = cFeature.view(batchSize, -1)
        predictions = self.linearSpeakerClassifier(cFeature)

        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)

        return loss, acc


class PhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder,
                 nLayers=1):

        super(PhoneCriterion, self).__init__()
        if nLayers == 1:
            self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones)
        else:
            outLayers = [nn.Linear(dimEncoder, nPhones)]
            for l in range(nLayers - 1):
                outLayers.append(nn.ReLU())
                outLayers.append(nn.Linear(nPhones, nPhones))
            self.PhoneCriterionClassifier = nn.Sequential(*outLayers)

        self.lossCriterion = nn.CrossEntropyLoss()
        self.onEncoder = onEncoder

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        if self.onEncoder:
            predictions = self.getPrediction(otherEncoded)
        else:
            predictions = self.getPrediction(cFeature)
        predictions = predictions.view(-1, predictions.size(2))
        label = label.view(-1)
        loss = self.lossCriterion(predictions, label).view(1, -1)
        acc = (predictions.max(1)[1] == label).double().mean().view(1, -1)
        return loss, acc

    def getPrediction(self, cFeature):
        batchSize, seqSize = cFeature.size(0), cFeature.size(1)
        cFeature = cFeature.contiguous().view(batchSize * seqSize, -1)
        output = self.PhoneCriterionClassifier(cFeature)
        return output.view(batchSize, seqSize, -1)


class CTCPhoneCriterion(BaseCriterion):

    def __init__(self, dimEncoder, nPhones, onEncoder, nLayers=1):

        super(CTCPhoneCriterion, self).__init__()
        if nLayers == 1:
            self.PhoneCriterionClassifier = nn.Linear(dimEncoder, nPhones + 1)
        else:
            outLayers = [nn.Linear(dimEncoder, nPhones + 1)]
            for l in range(nLayers - 1):
                outLayers.append(nn.ReLU())
                outLayers.append(nn.Linear(nPhones + 1, nPhones + 1))
            self.PhoneCriterionClassifier = nn.Sequential(*outLayers)
        self.lossCriterion = nn.CTCLoss(blank=nPhones, zero_infinity=True)
        self.onEncoder = onEncoder
        if onEncoder:
            raise ValueError("On encoder version not implemented yet")
        self.BLANK_LABEL = nPhones

    def getPrediction(self, cFeature):
        B, S, H = cFeature.size()
        cFeature = cFeature.contiguous().view(B*S, H)
        return self.PhoneCriterionClassifier(cFeature).view(B, S, -1)

    def forward(self, cFeature, otherEncoded, label):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature)
        label = label.to(predictions.device)
        label,  sizeLabels = collapseLabelChain(label)

        avgPER = 0.
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        targetSizePred = torch.ones(B, dtype=torch.int64,
                                    device=predictions.device) * S
        loss = self.lossCriterion(predictions, label,
                                  targetSizePred, sizeLabels).view(1, -1)

        return loss, avgPER * torch.ones(1, 1, device=loss.device)


class ModelCriterionCombined(torch.nn.Module):
    def __init__(self, model, criterion):
        super(ModelCriterionCombined, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, data, label):
        c_feature, encoded_data, label = self.model(data, label)
        loss, acc = self.criterion(c_feature, encoded_data, label)
        return loss, acc
