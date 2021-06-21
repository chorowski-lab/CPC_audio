# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch

import math

import time

from cpc.segm.hier_fast import FastHierarchicalSegmentationLayer, FastSegmentationLengthRestoreLayer
from cpc.segm.fast_segm_torch_conversions import padTens3ValueSetToLength

###########################################
# Networks
###########################################


class IDModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm",
                 noLastRelu=False):  # in original there was Relu there - same

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

        self.noLastRelu = noLastRelu

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = self.batchNorm4(self.conv4(x))
        #print("lastRelu:", self.lastRelu)
        #xlen = torch.sqrt(torch.clamp((x*x).sum(dim=-1), min=0))
        #print(f"x len: {(xlen.min(), xlen.mean(), xlen.max())}")
        if not self.noLastRelu:
            x = F.relu(x)
        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.dimEncoded = dimEncoded
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


# this uses dimOutput for hidden state dim - ok
class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimOutput,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            x = torch.flip(x, [1])
        return x


class NoAr(nn.Module):

    def __init__(self, *args):
        super(NoAr, self).__init__()

    def forward(self, x):
        return x


class BiDIRARTangled(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################

def seDistancesToCentroidsCpy(vecs, centroids, doNorm=False):
    #print(torch.square(centroids).sum(1).view(1,-1).shape, torch.square(vecs).sum(1).view(-1,1).shape, torch.matmul(vecs, centroids.T).shape)
    if len(vecs.shape) == 2:
        vecs = vecs.view(1, *(vecs.shape))
    B = vecs.shape[0]
    N = vecs.shape[1]
    k = centroids.shape[0]
    # vecs: B x L x Dim
    # centroids: k x Dim
    if doNorm:
        vecLengths = torch.sqrt((vecs*vecs).sum(-1))
        vecs = vecs / vecLengths.view(B, N, 1)
        centrLengths = torch.sqrt((centroids*centroids).sum(-1))
        centroids = centroids / centrLengths.view(k, 1)
        # print(vecLengths)
        # print(centrLengths)
        # vecLengths2 = (vecs*vecs).sum(-1)
        # print(f'vec lengths after norm from {vecLengths2.min().item()} to {vecLengths2.max().item()}')
        # centrLengths2 = (centroids*centroids).sum(-1)
        # print(f'center lengths after norm from {centrLengths2.min().item()} to {centrLengths2.max().item()}')
    # print(torch.square(centroids).sum(1).view(1, 1, -1).shape, torch.square(vecs).sum(-1).view(B, N, 1).shape,
    #     (vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1).shape,
    #     vecs.view(B, N, 1, -1).shape, centroids.view(1, 1, k, -1).shape)
    return torch.square(centroids).sum(1).view(1, 1, -1) + torch.square(vecs).sum(-1).view(B, N, 1) \
        - 2*(vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1)  #torch.matmul(vecs, centroids.T)


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR, 
                 perGPUbatchSize,
                 fcmSettings=None):

        super(CPCModel, self).__init__()
        self.normalBatchSize = perGPUbatchSize  # needed in some cases because DataParallel enforces absurd constraints on returned dim
        self.gEncoder = encoder
        self.gAR = AR
        self.fcm = fcmSettings is not None
        print(f'--------- FCM: {self.fcm} ----------')
        self.fcmDebug = False
        self.doing_push_loss_or_push_after = False
        #self.fcmReal = False
        self.hierARshorten = None  # will be set below if needed; otherwise need to set None
        self.VQpushEncCenterWeightOnTopConv = None
        self.VQpushEncCenterWeightOnlyAR = None
        self.VQpushEncCenterWeightOnlyCriterion = None
        self.VQpushCtxCenterWeight = None
        self.VQgradualStart = None
        self.hierARgradualStart = None
        self.modelLengthInARsimple = False
        self.modelLengthInARpredDep = None  # num preds if do
        self.shrinkEncodingsLengthDims = False
        self.showLengthsInCtx = False
        if self.fcm:
            self.fcmDebug = False   #True
            # self.fcmReal = fcmSettings["FCMproject"]
            self.shrinkEncodingsLengthDims = fcmSettings["shrinkEncodingsLengthDims"]
            self.showLengthsInCtx = fcmSettings["showLengthsInCtx"]
            self.modelLengthInARsimple = fcmSettings["modelLengthInARsimple"]
            self.modelLengthInARpredDep = fcmSettings["modelLengthInARpredDep"]  # num preds if do
            self.hierARshorten = fcmSettings["hierARshorten"]
            self.hierARgradualStart = fcmSettings["hierARgradualStart"]
            self.hierARmergePrior = fcmSettings["hierARmergePrior"]
            self.VQpushEncCenterWeightOnTopConv = fcmSettings["VQpushEncCenterWeightOnTopConv"]
            self.VQpushEncCenterWeightOnlyAR = fcmSettings["VQpushEncCenterWeightOnlyAR"]
            self.VQpushEncCenterWeightOnlyCriterion = fcmSettings["VQpushEncCenterWeightOnlyCriterion"]
            self.VQpushCtxCenterWeight = fcmSettings["VQpushCtxCenterWeight"]
            self.VQgradualStart = fcmSettings["VQgradualStart"]
            # caution - need to set args.hiddenGar to same as args.FCMprotos
            self.pushLossWeightEnc = fcmSettings["pushLossWeightEnc"]
            self.pushLossWeightCtx = fcmSettings["pushLossWeightCtx"]
            self.pushLossLinear = fcmSettings["pushLossLinear"]
            self.pushLossGradualStart = fcmSettings["pushLossGradualStart"]
            self.pushLossProtosMult = fcmSettings["pushLossProtosMult"]
            self.pushLossCenterNorm = fcmSettings["pushLossCenterNorm"]
            self.pushLossPointNorm = fcmSettings["pushLossPointNorm"]  # can be set only if centerNorm
            self.pushLossNormReweight = fcmSettings["pushLossNormReweight"]
            print("reweight:", self.pushLossNormReweight)
            if self.pushLossPointNorm:
                assert self.pushLossCenterNorm
            self.numProtos = fcmSettings["numProtos"]
            if self.pushLossWeightEnc is not None or self.pushLossWeightCtx is not None \
                or self.VQpushEncCenterWeightOnlyCriterion is not None or self.VQpushCtxCenterWeight is not None:

                self.doing_push_loss_or_push_after = True  # to be used in train
            else:
                self.doing_push_loss_or_push_after = False 
            
            self.reprDim = encoder.getDimOutput()
            ###print("----------------------Adding protos as parameter---------------------")
            # seems it's needed to also add requires_grad on the tensor step
            ###self.protos = nn.Parameter(torch.randn((self.numProtos, self.reprDim), requires_grad=True) / (5. * math.sqrt(self.reprDim)), requires_grad=True)  # TODO check


    # epochNrs: (current, total)
    def forward(self, arg1, arg2, arg3, arg4, givenCenters, epochNrs, calcPushLoss, onlyConv):
        # [!] OK, DataParallel splits in same way at both forward stages when push
        #print("::", arg1.shape, arg2.shape, calcPushLoss)
        epochNow_, epochAll_ = map(float,epochNrs)
        if not calcPushLoss:
            batchData, label, labelPhonePerGPU, maxSegmentCost = arg1, arg2, arg3, arg4
            
            encodedData = self.gEncoder(batchData).permute(0, 2, 1)

            if self.shrinkEncodingsLengthDims:
                if self.modelLengthInARsimple:
                    encodedData = encodedData[:,:,:-2]
                elif self.modelLengthInARpredDep is not None:
                    encodedData = encodedData[:,:,:-self.modelLengthInARpredDep]
            
            pureEncoded = encodedData.clone()
            encodedData = encodedData.clone()
            if onlyConv:
                return encodedData
            if givenCenters is not None and self.VQpushEncCenterWeightOnTopConv and (self.VQgradualStart is None or epochNow_ >= self.VQgradualStart):
                coeffOnTopConv = self.VQpushEncCenterWeightOnTopConv if self.VQgradualStart is None \
                    else self.VQpushEncCenterWeightOnTopConv*(max(epochNow_-self.VQgradualStart,0.)/max(epochAll_-self.VQgradualStart,1))
                encodedData = self._FCMlikeBelong(encodedData, givenCenters, None, None, None, coeffOnTopConv)

            #print("!", encodedData.shape)
            # TODO check if shape is like I think it is
            baseEncDim = encodedData.shape[-1]
            if self.fcmDebug:
                print('--------------------------------------forward fcmDebug')
                print(f'baseEncDim {baseEncDim}')
                print(f'epochNrs: {epochNrs}')
            
            #print("!!", encodedData.shape)
            if self.fcmDebug:
                print(f'enc data shape before AR: {encodedData.shape}')
            
            encForCfeature = encodedData
            if self.hierARshorten is not None:
                #--t0 = time.time()
                if self.hierARgradualStart is None:
                    shortening = self.hierARshorten
                else:
                    shortening = 1. + ((self.hierARshorten - 1.) * (max(epochNow_-self.hierARgradualStart, 0.)/max(epochAll_-self.hierARgradualStart,1.)))
                    # before chosen start epoch will just not shorten (will give 1 as shortening)
                    shortening = max(1., shortening)
                #print(f"SHORTENING: {shortening}; epoch data: {epochNow_} / {epochAll_}, start {self.hierARgradualStart}")
                lengthSumToObtain, _ = FastHierarchicalSegmentationLayer\
                    .getKforGivenShorteningAndShape(encodedData.shape, shortening)
                encForCfeature, segmDictTens, shrinkIndices_, lengths_, numsInLinesC0_, numsInLinesC1_, maxInLine_, encShapeTens_, segmCostForWantedLengthTens_, actualSegmK_ = \
                    FastHierarchicalSegmentationLayer.apply(encodedData, maxSegmentCost, lengthSumToObtain, 20, 5)
                #print("!!!", encShapeTens_)
                #--t1 = time.time()
                #--print(f"hier 1 time: {t1 - t0}; lengthSumToObtain {lengthSumToObtain}")
                #--print(f"shape segmented: {encForCfeature.shape}")
            else:
                segmDictTens = None
                segmCostForWantedLengthTens_ = None
                actualSegmK_ = None
            if givenCenters is not None and self.VQpushEncCenterWeightOnlyAR and (self.VQgradualStart is None or epochNow_ >= self.VQgradualStart):
                coeffOnlyAR = self.VQpushEncCenterWeightOnlyAR if self.VQgradualStart is None \
                    else self.VQpushEncCenterWeightOnlyAR*(max(epochNow_-self.VQgradualStart, 0.)/max(epochAll_-self.VQgradualStart,1.))
                encForCfeature = self._FCMlikeBelong(encForCfeature, givenCenters, None, None, None, coeffOnlyAR)
            if self.shrinkEncodingsLengthDims:
                if self.modelLengthInARsimple:
                    encForCfeature = torch.cat([encForCfeature, torch.zeros(1,1,1).cuda().repeat(encForCfeature.shape[0], encForCfeature.shape[1], 2)], dim=-1)  # append 0s at the end to make dim ok
                    #at the end encodedData = torch.cat([encodedData, torch.zeros(1,1,1).cuda().repeat(encodedData.shape[0], encodedData.shape[1], 2)], dim=-1)  # append 0s at the end to make dim ok
                elif self.modelLengthInARpredDep is not None:
                    encForCfeature = torch.cat([encForCfeature, torch.zeros(1,1,1).cuda().repeat(encForCfeature.shape[0], encForCfeature.shape[1], self.modelLengthInARpredDep)], dim=-1)  # append 0s at the end to make dim ok
                    #at the end encodedData = torch.cat([encodedData, torch.zeros(1,1,1).cuda().repeat(encodedData.shape[0], encodedData.shape[1], self.modelLengthInARpredDep)], dim=-1)  # append 0s at the end to make dim ok
            cFeature = self.gAR(encForCfeature)

            if self.fcmDebug:
                print(f'ctx data shape just after AR: {cFeature.shape}')

            predictedLengths = None
            if self.modelLengthInARsimple:
                predictedLengths = cFeature[:,:,-1]
                if self.showLengthsInCtx:
                    #cFeature[:,:,-2] = 0.  # only zero -2, in -1 there are lengths already
                    cFeature = torch.cat([
                            cFeature[:,:,:-2], 
                            torch.zeros_like(cFeature[:,:,-2]).view(cFeature.shape[0], cFeature.shape[1], 1), 
                            predictedLengths.view(cFeature.shape[0], cFeature.shape[1], 1)],
                        dim=-1)
                else:
                    #cFeature[:,:,-2:] = 0.
                    cFeature = torch.cat([cFeature[:,:,:-2], torch.zeros_like(cFeature[:,:,-2:])], dim=-1)
            elif self.modelLengthInARpredDep is not None:
                predictedLengths = cFeature[:,:,-self.modelLengthInARpredDep:]
                if self.showLengthsInCtx:
                    pass  # length already in ctx
                else:
                    #cFeature[:,:,-self.modelLengthInARpredDep:] = 0.
                    cFeature = torch.cat([cFeature[:,:,:-self.modelLengthInARpredDep], predictedLengths], dim=-1)
            if self.hierARshorten is not None:  # TODO here or at the end, unsure
                #--t0 = time.time()
                #cFeature = HierarchicalSegmentationRestoreLengthLayer.apply(cFeature, segmDictTens)
                cFeature = FastSegmentationLengthRestoreLayer.apply(
                    cFeature, numsInLinesC0_, numsInLinesC1_, shrinkIndices_, maxInLine_, torch.Size(encShapeTens_))
                #--t1 = time.time()
                #--print(f"hier 2 time: {t1 - t0}")
                #segmDictTens = segmDictTens.cuda()  # for return  TODO check if all returned tensors need to be on the same device, if so, move also segmCostForWantedLengthTens_ and actualSegmK_

            if self.fcmDebug:
                print(f'ctx shape returned {cFeature.shape}')
                print(f'enc shape returned {encodedData.shape}')

            pushLoss = torch.full((1,), baseEncDim, dtype=int).cuda() if self.doing_push_loss_or_push_after else None

            #--t01 = time.time()
            #print(f"normalBatchSize: {self.normalBatchSize} x {encodedData.shape[1]}")
            if segmDictTens is not None:
                segmDictTens = padTens3ValueSetToLength(segmDictTens, self.normalBatchSize*encodedData.shape[1])  
                # ^ needed because of torch being extremely indiscriminate, making things unnecessary user unfriendly
                #   (with dataParallel whole shape needs to be the same and not only dim0); also, need to pad to max possible size
                #   as it can happen that GPU0 batch has different dim0 than GPU0 one and dataParallel will be even more annoying
                segmDictTens = segmDictTens.view(1,-1).cuda()

            if segmCostForWantedLengthTens_ is not None:
                segmCostForWantedLengthTens_ = segmCostForWantedLengthTens_.view(1,-1).cuda()

            if actualSegmK_ is not None:
                actualSegmK_ = actualSegmK_.view(1,-1).cuda()

            if labelPhonePerGPU is not None:
                # ugly things below needed because dataParallel is hopeless as described several lines above
                labelPhonePerGPU2 = torch.zeros(self.normalBatchSize, encodedData.shape[1], dtype=labelPhonePerGPU.dtype).cuda()
                labelPhonePerGPU2[:labelPhonePerGPU.shape[0], :labelPhonePerGPU.shape[1]] = labelPhonePerGPU
                labelPhonePerGPU = labelPhonePerGPU2
                labelPhonePerGPU = labelPhonePerGPU.view(1,*(labelPhonePerGPU.shape))
            #--t02 = time.time()
            #--print(f"part of additional time lost because of DataParallel: {t02-t01}")

            if self.shrinkEncodingsLengthDims:
                if self.modelLengthInARsimple:
                    encodedData = torch.cat([encodedData, torch.zeros(1,1,1).cuda().repeat(encodedData.shape[0], encodedData.shape[1], 2)], dim=-1)  # append 0s at the end to make dim ok
                elif self.modelLengthInARpredDep is not None:
                    encodedData = torch.cat([encodedData, torch.zeros(1,1,1).cuda().repeat(encodedData.shape[0], encodedData.shape[1], self.modelLengthInARpredDep)], dim=-1)  # append 0s at the end to make dim ok

            return cFeature, predictedLengths, encodedData, pureEncoded, label, labelPhonePerGPU, pushLoss, segmDictTens, segmCostForWantedLengthTens_, actualSegmK_

        else:

            if givenCenters is None:
                #print(f"***NONE CENTERS, ep. {epochNrs[0]}")
                return torch.zeros(1).cuda(), torch.zeros(1,self.numProtos).cuda(), arg1, arg2

            # had to do it like that, as couldn't return tensors and later check grad as 
            # DataParallel spoiled everything making a new concatenated tensor
            cFeature, encodedData, cFeatureForPushLoss, encodedDataForPushLoss = arg1, arg2, arg3, arg4
            baseEncDim = self.gEncoder.getDimOutput()
            
            xLoss = torch.zeros(1).cuda()
            usedCounts = torch.zeros(1).cuda()
            cFeatureOut = cFeature
            encodedDataOut = encodedData

            if self.pushLossWeightEnc is not None or self.pushLossWeightCtx is not None\
                and (self.pushLossGradualStart is None or epochNow_ >= self.pushLossGradualStart):

                # needs enc to go through LSTM; shouldn't have non-loss pushing set
                pushLoss = torch.zeros(1).cuda().sum()
                # there was a bug with just [:baseEncDim] here, but with only-pushloss config should change anything
                encodedDataPushLossPart = encodedDataForPushLoss[:, :, :baseEncDim]  #.clone()
                ctxDataPushLossPart = cFeatureForPushLoss[:, :, :baseEncDim]  #.clone()
                # [!] need to also clone original tensors, otherwise gradient there would also have pushLoss part
                #     ^ BUT NEED TO DO SO IN TRAIN BECAUSE OF DATAPARALLEL
                if self.pushLossGradualStart is not None:
                    #currentEpoch, allEpochs = epochNrs
                    weightMult = max(epochNow_ - self.pushLossGradualStart,0.) / max(epochAll_ - self.pushLossGradualStart, 1.)
                else:
                    weightMult = 1.
                protoUsedCounts1 = 0
                protoUsedCounts2 = 0
                if self.pushLossWeightEnc is not None:
                    lossPart1, protoUsedCounts1 = self._FCMlikeBelong(encodedDataPushLossPart, givenCenters, None, None, weightMult * self.pushLossWeightEnc)
                    pushLoss = pushLoss + lossPart1
                if self.pushLossWeightCtx is not None:
                    lossPart2, protoUsedCounts2 = self._FCMlikeBelong(ctxDataPushLossPart, givenCenters, None, None, weightMult * self.pushLossWeightCtx)
                    pushLoss = pushLoss + lossPart2
                # https://discuss.pytorch.org/t/dataparallel-only-supports-tensor-output/34519
                
                # [!] need to do retain_grad on returned values of cFeaturePushLoss, ctxDataPushPart, encodedDataPushLoss, encodedDataPushPart
                #     in train.py as those act as different access points
                #     and doing it here has no effect on them there
                # [!] can't even check grad as the copy is returned - had to make this 2-variant forward
                
                xLoss += pushLoss

                usedCounts = usedCounts + protoUsedCounts1 + protoUsedCounts2
                #print(x.shape)
                #print(":::::", givenCenters.shape, protoUsedCounts1.shape, protoUsedCounts1)

            # VQ push will not be used for linsep with those two - as this is "only for criterion VQ"
            if self.VQpushEncCenterWeightOnlyCriterion is not None and (self.VQgradualStart is None or epochNow_ >= self.VQgradualStart):
                encodedDataPushPart = encodedData[:, :, :baseEncDim]  #.clone()
                coeffOnlyCritEnc = self.VQpushEncCenterWeightOnlyCriterion if self.VQgradualStart is None \
                    else self.VQpushEncCenterWeightOnlyCriterion*(max(epochNow_-self.VQgradualStart,0.)/max(epochAll_-self.VQgradualStart,1))
                encodedDataOut = self._FCMlikeBelong(encodedDataPushPart, givenCenters, None, None, None, coeffOnlyCritEnc)

            if self.VQpushCtxCenterWeight is not None and (self.VQgradualStart is None or epochNow_ >= self.VQgradualStart):
                ctxDataPushPart = cFeature[:, :, :baseEncDim]  #.clone()
                coeffOnlyCritCtx = self.VQpushCtxCenterWeight if self.VQgradualStart is None \
                    else self.VQpushCtxCenterWeight*(max(epochNow_-self.VQgradualStart,0.)/max(epochAll_-self.VQgradualStart,1))
                #print(";", coeffOnlyCritCtx)
                cFeatureOut = self._FCMlikeBelong(ctxDataPushPart, givenCenters, None, None, None, coeffOnlyCritCtx)
            
            return xLoss, usedCounts, cFeatureOut, encodedDataOut  #pushLoss  #torch.full((1,), baseEncDim, dtype=int).cuda(), pushLoss
        # else:
        #     pushLoss = None

        # return cFeature, encodedData, label, pushLoss

    #@staticmethod
    def _FCMlikeBelong(self, points, centers, m=None, pushDeg=None, pushLossWeight=None, pushDegWithCenterDetach=None):  # for pushDeg no FCM; pushDeg OR m? TODO BUT COULD ALSO TRY THAT WEIGHTED PUSH

        # probably didn't help too much, but maybe a bit
        if (pushLossWeight is not None or pushDegWithCenterDetach is not None) and self.pushLossCenterNorm:
            pointsLens = torch.sqrt(torch.clamp((points*points).sum(dim=-1), min=0))  #.mean()
            centersLens = torch.sqrt(torch.clamp((centers*centers).sum(dim=-1), min=0))  #.mean()

            #print(points.shape, centers.shape, pointLens.shape, centersLens.shape, pointLens.view(*(pointLens.shape), 1).shape)
            #centers = (centers / (centersLens.view(-1,1))) #* pointLens  # avg 5 times shorter
            #pointLens = (points / pointLens.view(*(pointLens.shape), 1))
            #print("@@@@@@@@@@@@@", pointLens, centersLens)
            if not self.pushLossPointNorm:
                pointsLensAvg = pointsLens.mean()
                centers = (centers / torch.clamp(centersLens.view(-1,1), min=1)) * pointsLensAvg
            else:
                centers = centers / torch.clamp(centersLens.view(-1,1), min=1)
                points = points / torch.clamp(pointsLens.view(*(points.shape[:-1]),1), min=1)
                if self.pushLossNormReweight and pushLossWeight is not None:  # for similar pushing weight as without norm
                    # have to check for none, as in pushDegWithCenterDetach also entering here
                    pushLossWeight *= pointsLens.mean()
            # if we make only centers much shorter and encodings not, encodings will just be pushed to 0, each similarly

        if self.pushLossProtosMult is None:  
            distsSq = seDistancesToCentroidsCpy(points, centers)
            distsSq = torch.clamp(distsSq, min=0)
            dists = torch.sqrt(distsSq)  
        else:  # only to be used with protos, not when possible future k-means
            # VQ-VAE-commitment-loss-weight - like
            assert pushLossWeight is not None

            distsSq1 = seDistancesToCentroidsCpy(points, centers.clone().detach())
            distsSq1 = torch.clamp(distsSq1, min=0)
            dists1 = torch.sqrt(distsSq1)  
            distsSq2 = seDistancesToCentroidsCpy(points.clone().detach(), centers)
            distsSq2 = torch.clamp(distsSq2, min=0)
            dists2 = torch.sqrt(distsSq2)
            # just sum distances, as later only linear stuff on this is made to obtain loss
            # and this sum preserves order ((1+self.pushLossProtosMult) * distances)
            # actual values are not important, only gradients which will be correct
            distsSq = distsSq1 + self.pushLossProtosMult * distsSq2
            dists = dists1 + self.pushLossProtosMult * dists2
        # dists: B x N x k

        k = dists.shape[2]
        N = points.shape[1]
        B = points.shape[0]

        #print(dists.shape)

        if m is not None:
            distsRev = 1. / torch.clamp(dists, min=0.00000001)  #torch.maximum(dists, torch.tensor(0.00000001).cuda())
            # print("________________________")
            # print(dists, distsRev)
            # print('========================')
            #print(distsRev.shape, k)
            distsRev3dim = distsRev.view(*(distsRev.shape[:2]), 1, k)
            dists3dim = dists.view(*dists.shape, 1)
            distsDiv = dists3dim * distsRev3dim
            powed = torch.pow(distsDiv, 2./(m-1.))  #1./(m-1.))  #2./(m-1.))  TODO avoiding nans
            # print(distsDiv)
            # print(powed)
            denomin = torch.sum(powed, (-1))
            # print(denomin)
            return 1. / torch.clamp(denomin, min=0.00000001)  #torch.maximum(denomin, torch.tensor(0.00000001).cuda())

        elif pushDeg is not None:  # centerpush
            closest = dists.argmin(-1)
            # print(points.shape, closest.shape, centers[closest].view(N, -1).shape)
            diffs = centers[closest].view(B, N, -1) - points
            return pushDeg * diffs + points
            # print(diffs.shape)

        elif pushLossWeight is not None:

            if self.pushLossLinear:
                dst = dists
            else:
                dst = distsSq
            
            minDistsData = dst.min(dim=2)
            minDistsValues = minDistsData.values
            minDistsIndices = minDistsData.indices
            #print(minDistsIndices)
            indices, indicesCounts = torch.unique(minDistsIndices, return_counts=True)
            closestCounts = torch.zeros(self.numProtos, dtype=int).cuda()
            closestCounts[indices] += indicesCounts
            #print(indicesCounts)
            mean = minDistsValues.mean()
            pushLoss = mean * pushLossWeight
            #print("--->!", pushLoss, pushLoss.shape, mean.shape)
            #res = pushLoss
            return pushLoss, closestCounts.view(1,-1)  # view because of dataparallel

        elif pushDegWithCenterDetach is not None:

            closest = dists.argmin(-1)
            closestCenters = centers[closest].view(B, N, -1)
            pushedPoints = points + (pushDegWithCenterDetach*closestCenters - pushDegWithCenterDetach*points).detach()  # push*closestCenters + (1-push)*points in forward
            # [!] this doesn't normalize, here, but normalization is done above also in this case
            #     if centers are normalized (in centermodel), we have problem here
            #     so we need to do it like that v
            #print("===", closestCenters, )
            if self.pushLossCenterNorm and self.pushLossPointNorm:
                pushedPoints *= pointsLens.view(*(pointsLens.shape), 1)  # restore original lengths not to collapse those even if pushing to cosine-closest in cosine-way
            return pushedPoints

        else:
            assert False



class CPCModelNullspace(nn.Module):

    def __init__(self,
                 cpc,
                 nullspace):

        super(CPCModelNullspace, self).__init__()
        self.cpc = cpc
        self.nullspace = nn.Linear(nullspace.shape[0], nullspace.shape[1], bias=False)
        self.nullspace.weight = nn.Parameter(nullspace.T)
        self.gEncoder = self.cpc.gEncoder


    def forward(self, batchData, label):
        cFeature, encodedData, label = self.cpc(batchData, label)
        cFeature = self.nullspace(cFeature)
        encodedData = self.nullspace(encodedData)
        return cFeature, encodedData, label


class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), \
            torch.cat(outEncoded, dim=2), label
