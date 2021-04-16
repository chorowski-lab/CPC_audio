# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch

import math

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
                 normMode="layerNorm"):

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

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
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
                 fcmSettings=None):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.fcm = fcmSettings is not None
        print(f'--------- FCM: {self.fcm} ----------')
        self.fcmDebug = False
        if self.fcm:
            self.fcmDebug = False   #True
            # caution - need to set args.hiddenGar to same as args.FCMprotos
            self.reprsConcat = fcmSettings["reprsConcat"]
            self.reprsConcatNormSumsNotLengths = fcmSettings["reprsConcatNormSumsNotLengths"]
            self.numProtos = fcmSettings["numProtos"]
            self.mBeforeAR = fcmSettings["mBeforeAR"]
            self.leftProtos = fcmSettings["leftProtos"]  # either m OR pushDeg
            self.pushDegFeatureBeforeAR = fcmSettings["pushDegFeatureBeforeAR"]
            self.mAfterAR = fcmSettings["mAfterAR"]
            self.pushDegCtxAfterAR = fcmSettings["pushDegCtxAfterAR"]
            self.pushDegAllAfterAR = fcmSettings["pushDegAllAfterAR"]
            
            # encoder.getDimOutput() is what was fed to AR and had to be same as AR.getDimOutput() for sim
            # now replacing encoder.getDimOutput() with self.numProto dims in case with m, leaving with pushDeg
            assert self.mBeforeAR is None or self.mAfterAR is None
            assert self.pushDegCtxAfterAR is None or self.pushDegAllAfterAR  is None  # if push both, use only the latter
            # if self.mBeforeAR is not None or self.mAfterAR is not None:
            #     assert self.pushDegFeatureBeforeAR is None and self.pushDegCtxAfterAR is None and self.pushDegAllAfterAR is None
            # if self.mBeforeAR is not None:
            #     assert self.numProtos == AR.getDimOutput()
            #if self.leftProtos is not None and self.leftProtos > 0:
            if self.mAfterAR is not None:
                pass  # in this case needed to pass correct AR output (changed) dim to criterion
            # elif self.pushDegFeatureBeforeAR is not None or self.pushDegCtxAfterAR is not None or self.pushDegAllAfterAR is not None:
            #     # no dim change in those cases
            #     assert encoder.getDimOutput() == AR.getDimOutput()
            #     assert self.leftProtos is None
            self.reprDim = encoder.getDimOutput()
            print("----------------------Adding protos as parameter---------------------")
            # seems it's needed to also add requires_grad on the tensor step
            self.protos = nn.Parameter(torch.randn((self.numProtos, self.reprDim), requires_grad=True) / math.sqrt(self.reprDim), requires_grad=True)  # TODO check

    #@staticmethod
    # TODO option to normalize sums and not lengths
    def gradualTrainingNormalize(self, part1, part2, epochNrs):

        currentEpoch, totalEpochs = epochNrs

        # # TODO maybe add option to specify moment when part2 starts to learn
        partOfEpochs = (currentEpoch + 1) / (totalEpochs + 1)  # +-1, don't want 0-division anywhere
        # if partOfEpochs < 0.25:  # this would mean a possible sudden jump in lengths when this ends
        #     return part1, 0.*part2
        if self.fcmDebug:
            print(f'PART OF EPOCHS: {partOfEpochs}')
        
        # partOfEpochs = (currentEpoch - 0.25) / (totalEpochs - 0.25)

        # [!] clamping to 1 so that shorter vectors are not normalized - to prevent problems with initial training
        #     TODO think if do this also for part 2 - if not, would increase length of stuff like (0.3, 0.3, 0.4)
        #          (as length is always <1 after fcm, only LSTM can increase) and also of stuff when we cut part of noise-assumed protos
        #          --> so, for now clamping linear dists
        part1lengthsNoClamp = torch.sqrt((part1*part1).sum(-1)) if not self.reprsConcatNormSumsNotLengths else (torch.abs(part1)).sum(-1)
        if self.fcmDebug:
            print(f'part1lengthsNoClamp: {(part1lengthsNoClamp.min().item(), part1lengthsNoClamp.mean().item(), part1lengthsNoClamp.max().item())}')
        part1lengths = torch.clamp(part1lengthsNoClamp, min=1.)  #min=0.00000000001)
        part1 = part1 / part1lengths.view(*(part1lengths.shape), 1)
        part1 *= (1. - partOfEpochs)  # [!] length is upper bound by (1. - partOfEpochs) after that
        part1lengthsFinal = (part1lengthsNoClamp / part1lengths) * (1. - partOfEpochs)  # [!] length is upper bound by (1. - partOfEpochs) after that
        if self.fcmDebug:
            print(f'part1lengthsFinal: {(part1lengthsFinal.min().item(), part1lengthsFinal.mean().item(), part1lengthsFinal.max().item())}')
        part2lengthsNoClamp = torch.sqrt((part2*part2).sum(-1)) if not self.reprsConcatNormSumsNotLengths else (torch.abs(part2)).sum(-1)
        if self.fcmDebug:
            print(f'part2lengthsNoClamp: {(part2lengthsNoClamp.min().item(), part2lengthsNoClamp.mean().item(), part2lengthsNoClamp.max().item())}')
        part2lengths = torch.clamp(part2lengthsNoClamp, min=1.)  #min=0.00000000001)
        if self.fcmDebug:
            print(f'part2lengths: {(part2lengths.min().item(), part2lengths.mean().item(), part2lengths.max().item())}')
        #part2sums = torch.clamp(part2.sum(-1), min=1.)
        # [!] after below, length is <= : big () is <= partOfEpochs, whole thing also
        part2 = part2 * (part1lengthsFinal * partOfEpochs / (1. - partOfEpochs)).view(*(part2lengths.shape), 1) / part2lengths.view(*(part2lengths.shape), 1)
        #part2 *= partOfEpochs

        return part1, part2

    # epochNrs: (current, total)
    def forward(self, batchData, label, epochNrs):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        #print("!", encodedData.shape)
        # TODO check if shape is like I think it is
        baseEncDim = encodedData.shape[-1]
        if self.fcmDebug:
            print('--------------------------------------forward fcmDebug')
            print(f'baseEncDim {baseEncDim}')
            print(f'epochNrs: {epochNrs}')
        if self.fcm:
            # could just invoke with some of those as None but this way it's more readable I guess
            if self.reprsConcat and self.mBeforeAR is not None:
                encodedDataM = CPCModel._FCMlikeBelong(encodedData, self.protos, self.mBeforeAR, None)
                if self.fcmDebug:
                    print(f'enc data m before construction')
                if self.pushDegFeatureBeforeAR is not None:
                    encodedDataPush = CPCModel._FCMlikeBelong(encodedData, self.protos, None, self.pushDegFeatureBeforeAR)
                    if self.fcmDebug:
                        print(f'enc data pushing before')
                else:
                    encodedDataPush = encodedData
                # normalize parts of it and train cluster-distance representations gradually
                # at the beginning, regular representations are most of final vec length,
                # so that prototypes and LSTM can be reasonably trained
                # and then there is transition to another (proto-dist-based) representation
                encodedDataPush, encodedDataM = self.gradualTrainingNormalize(encodedDataPush, encodedDataM, epochNrs)
                if self.fcmDebug:
                    print(f'enc data concat before')
                    lens1 = torch.sqrt((encodedDataPush*encodedDataPush).sum(-1))
                    sums1 = encodedDataPush.abs().sum(-1)
                    lens2 = torch.sqrt((encodedDataM*encodedDataM).sum(-1))  #encodedDataM.sum(-1)
                    sums2 = encodedDataM.abs().sum(-1)
                    print(f'push lengths min, avg, max {(lens1.min().item(), lens1.mean().item(), lens1.max().item())},',
                          f'push sums min, avg, max {(sums1.min().item(), sums1.mean().item(), sums1.max().item())},',
                          f'm lengths min, avg, max: {(lens2.min().item(), lens2.mean().item(), lens2.max().item())}',
                          f'm sums min, avg, max: {(sums2.min().item(), sums2.mean().item(), sums2.max().item())}')
                # TODO check
                encodedData = torch.cat([encodedDataPush, encodedDataM], dim=-1)
            elif self.mBeforeAR is not None:
                if self.pushDegFeatureBeforeAR is not None:  # can push before m, but not sure if makes sense
                    encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, None, self.pushDegFeatureBeforeAR)
                    if self.fcmDebug:
                        print(f'enc data pushing before')
                encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, self.mBeforeAR, None)
                if self.fcmDebug:
                    print(f'enc data m before construction')
            elif self.pushDegFeatureBeforeAR is not None:
                encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, None, self.pushDegFeatureBeforeAR)
                if self.fcmDebug:
                    print(f'enc data pushing before')
        #print("!!", encodedData.shape)
        if self.fcmDebug:
            print(f'enc data shape before AR: {encodedData.shape}')
        cFeature = self.gAR(encodedData)
        if self.fcmDebug:
            print(f'ctx data shape just after AR: {cFeature.shape}')
        if self.fcm:
            
            B = cFeature.shape[0]
            N = cFeature.shape[1]

            if not self.reprsConcat:

                if self.mBeforeAR is not None:
                    # normalization - as encodedData is normalized, making LSTM's life easier
                    # and not forcing it to figure normalization out itself, when 
                    # making good things bigger only seems good up to 1 and later weird stuff happens
                    cFeatureLengths = (cFeature).sum(-1)
                    cFeature = cFeature / torch.clamp(cFeatureLengths.view(B, N, 1), min=1.)  #min=0.00000000001)
                    # ^ changed to linear sum and clamping to 1 so that 
                    if self.fcmDebug:
                        print(f"m before's after-AR ctx norm")

                    # cut "transition" protos after LSTM, before CPC criterion
                    # only makes sense in mBeforeAR case
                    if self.leftProtos and self.leftProtos > 0:
                        encodedData = encodedData[:, :, :self.leftProtos]
                        cFeature = cFeature[:, :, :self.leftProtos]
                        if self.fcmDebug:
                            print(f"m before's protos cutting")
                
                elif self.mAfterAR is not None:
                    if self.pushDegCtxAfterAR is not None:
                        cFeature = CPCModel._FCMlikeBelong(cFeature, self.protos, None, self.pushDegCtxAfterAR)
                        if self.fcmDebug:
                            print(f'ctx pushing after, before m after')

                    if self.pushDegAllAfterAR is not None:
                        encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, None, self.pushDegAllAfterAR)
                        cFeature = CPCModel._FCMlikeBelong(cFeature, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f'enc and ctx pushing after, before m after')
                    encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, self.mAfterAR, None)
                    cFeature = CPCModel._FCMlikeBelong(cFeature, self.protos, self.mAfterAR, None)
                    if self.fcmDebug:
                        print(f'enc and ctx m after')

                # pushes after AR can be done, but need to be before m stuff, hence done above plus else here

                else:

                    if self.pushDegCtxAfterAR is not None:
                        cFeature = CPCModel._FCMlikeBelong(cFeature, self.protos, None, self.pushDegCtxAfterAR)
                        if self.fcmDebug:
                            print(f'ctx pushing after')

                    if self.pushDegAllAfterAR is not None:
                        encodedData = CPCModel._FCMlikeBelong(encodedData, self.protos, None, self.pushDegAllAfterAR)
                        cFeature = CPCModel._FCMlikeBelong(cFeature, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f'enc and ctx pushing after')

            else:

                if self.mBeforeAR is not None:
                    cFeaturePushPart = cFeature[:, :, :baseEncDim]
                    cFeatureMpart = cFeature[:, :, baseEncDim:]
                    # need to do pushing before norm
                    if self.pushDegCtxAfterAR is not None:
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegCtxAfterAR)
                        if self.fcmDebug:
                            print(f"m before's ctx push after with concat")
                    if self.pushDegAllAfterAR is not None:
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegAllAfterAR)
                        encodedDataPushPart = encodedData[:, :, :baseEncDim]
                        encodedDataMpart = encodedData[:, :, baseEncDim:]
                        encodedDataPushPart = CPCModel._FCMlikeBelong(encodedDataPushPart, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f"m before's all push after with concat")
                        encodedData = torch.cat([encodedDataPushPart, encodedDataMpart], dim=-1)
                    # this also normalizes m part after LSTM, similarly as when no repr concat
                    cFeaturePushPart, cFeatureMpart = self.gradualTrainingNormalize(cFeaturePushPart, cFeatureMpart, epochNrs)
                    # TODO check
                    if self.leftProtos and self.leftProtos > 0:
                        encodedDataPushPart = encodedData[:, :, :baseEncDim]
                        encodedDataMpart = encodedData[:, :, baseEncDim:]
                        encodedDataMpart = encodedDataMpart[:, :, :self.leftProtos]
                        cFeature = torch.cat([encodedDataPushPart, encodedDataMpart], dim=-1)
                        cFeatureMpart = cFeatureMpart[:, :, :self.leftProtos]
                        if self.fcmDebug:
                            print(f"m before's protos cutting with concat")
                    # TODO check
                    if self.fcmDebug:
                        print(f'ctx data concat after for m before')
                        lens1 = torch.sqrt((cFeaturePushPart*cFeaturePushPart).sum(-1))
                        sums1 = cFeaturePushPart.abs().sum(-1)
                        lens2 = torch.sqrt((cFeatureMpart*cFeatureMpart).sum(-1))  #encodedDataM.sum(-1)
                        sums2 = cFeatureMpart.abs().sum(-1)
                        print(f'push lengths min, avg, max {(lens1.min().item(), lens1.mean().item(), lens1.max().item())},',
                            f'push sums min, avg, max {(sums1.min().item(), sums1.mean().item(), sums1.max().item())},',
                            f'm lengths min, avg, max: {(lens2.min().item(), lens2.mean().item(), lens2.max().item())}',
                            f'm sums min, avg, max: {(sums2.min().item(), sums2.mean().item(), sums2.max().item())}')
                    cFeature = torch.cat([cFeaturePushPart, cFeatureMpart], dim=-1)
                
                elif self.mAfterAR is not None:  # here encodedData and cFeature are regular-dim at the beginning
                    cFeaturePushPart = cFeature
                    cFeatureMpart = CPCModel._FCMlikeBelong(cFeature, self.protos, self.mAfterAR, None)
                    # need to do pushing before norm
                    if self.pushDegCtxAfterAR is not None:
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegCtxAfterAR)
                        if self.fcmDebug:
                            print(f"m after's ctx push after with concat")
                    if self.pushDegAllAfterAR is not None:
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f"m after's all (ctx now) push after with concat")
                    cFeaturePushPart, cFeatureMpart = self.gradualTrainingNormalize(cFeaturePushPart, cFeatureMpart, epochNrs)
                    if self.fcmDebug:
                        print(f'ctx data concat after for m after')
                        lens1 = torch.sqrt((cFeaturePushPart*cFeaturePushPart).sum(-1))
                        sums1 = cFeaturePushPart.abs().sum(-1)
                        lens2 = torch.sqrt((cFeatureMpart*cFeatureMpart).sum(-1))  #encodedDataM.sum(-1)
                        sums2 = cFeatureMpart.abs().sum(-1)
                        print(f'push lengths min, avg, max {(lens1.min().item(), lens1.mean().item(), lens1.max().item())},',
                            f'push sums min, avg, max {(sums1.min().item(), sums1.mean().item(), sums1.max().item())},',
                            f'm lengths min, avg, max: {(lens2.min().item(), lens2.mean().item(), lens2.max().item())}',
                            f'm sums min, avg, max: {(sums2.min().item(), sums2.mean().item(), sums2.max().item())}')
                    cFeature = torch.cat([cFeaturePushPart, cFeatureMpart], dim=-1)
                    encodedDataPushPart = encodedData
                    encodedDataMpart = CPCModel._FCMlikeBelong(encodedData, self.protos, self.mAfterAR, None)
                    # need to do pushing before norm
                    if self.pushDegAllAfterAR is not None:
                        encodedDataPushPart = CPCModel._FCMlikeBelong(encodedDataPushPart, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f"m after's all (enc now) push after with concat")
                    encodedDataPushPart, encodedDataMpart = self.gradualTrainingNormalize(encodedDataPushPart, encodedDataMpart, epochNrs)
                    if self.fcmDebug:
                        print(f'enc data concat after for m after')
                        lens1 = torch.sqrt((encodedDataPushPart*encodedDataPushPart).sum(-1))
                        sums1 = encodedDataPushPart.abs().sum(-1)
                        lens2 = torch.sqrt((encodedDataMpart*encodedDataMpart).sum(-1))  #encodedDataM.sum(-1)
                        sums2 = encodedDataMpart.abs().sum(-1)
                        print(f'push lengths min, avg, max {(lens1.min().item(), lens1.mean().item(), lens1.max().item())},',
                            f'push sums min, avg, max {(sums1.min().item(), sums1.mean().item(), sums1.max().item())},',
                            f'm lengths min, avg, max: {(lens2.min().item(), lens2.mean().item(), lens2.max().item())}',
                            f'm sums min, avg, max: {(sums2.min().item(), sums2.mean().item(), sums2.max().item())}')
                    encodedData = torch.cat([encodedDataPushPart, encodedDataMpart], dim=-1)

                # ones below in separate else as in cases above need to do pushing before other things and not here at the end
                else:
                    if self.pushDegCtxAfterAR is not None:
                        cFeaturePushPart = cFeature[:, :, :baseEncDim]
                        cFeatureMpart = cFeature[:, :, baseEncDim:]
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegCtxAfterAR)
                        if self.fcmDebug:
                            print(f"regular ctx push after")
                        cFeature = torch.cat([cFeaturePushPart, cFeatureMpart], dim=-1)

                    if self.pushDegAllAfterAR is not None:
                        cFeaturePushPart = cFeature[:, :, :baseEncDim]
                        cFeatureMpart = cFeature[:, :, baseEncDim:]
                        cFeaturePushPart = CPCModel._FCMlikeBelong(cFeaturePushPart, self.protos, None, self.pushDegAllAfterAR)
                        cFeature = torch.cat([cFeaturePushPart, cFeatureMpart], dim=-1)
                        encodedDataPushPart = encodedData[:, :, :baseEncDim]
                        encodedDataMpart = encodedData[:, :, baseEncDim:]
                        encodedDataPushPart = CPCModel._FCMlikeBelong(encodedDataPushPart, self.protos, None, self.pushDegAllAfterAR)
                        if self.fcmDebug:
                            print(f"regular all push after")
                        encodedData = torch.cat([encodedDataPushPart, encodedDataMpart], dim=-1)

        if self.fcmDebug:
            print(f'ctx shape returned {cFeature.shape}')
            print(f'enc shape returned {encodedData.shape}')

        return cFeature, encodedData, label

    @staticmethod
    def _FCMlikeBelong(points, centers, m=None, pushDeg=None):  # for pushDeg no FCM; pushDeg OR m? TODO BUT COULD ALSO TRY THAT WEIGHTED PUSH

        distsSq = seDistancesToCentroidsCpy(points, centers)
        #print(distsSq.dtype)
        dists = distsSq  #torch.sqrt(distsSq)  TODO avoiding nans
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
            powed = torch.pow(distsDiv, 1./(m-1.))  #2./(m-1.))  TODO avoiding nans
            # print(distsDiv)
            # print(powed)
            denomin = torch.sum(powed, (-1))
            # print(denomin)
            res = 1. / torch.clamp(denomin, min=0.00000001)  #torch.maximum(denomin, torch.tensor(0.00000001).cuda())

        elif pushDeg is not None:  # centerpush
            #print(res.shape, centers.shape)
            #print(res)
            #print(points.view(points.shape[0], 1, points.shape[1]).shape, centers.view(1, centers.shape[0], centers.shape[1]).shape)
            # diffs = points.view(points.shape[0], 1, points.shape[1]) - centers.view(1, centers.shape[0], centers.shape[1])
            # #print(diffs.shape, res.shape)
            # resNotSummed = (1 - deg * res.view(*res.shape, 1)) * diffs + centers  #torch.matmul(res, diffs) #+ centers #torch.matmul(res, centers)
            # #print(resNotSummed.shape, res.shape)
            # resNotAvg = k*res.view(*res.shape, 1) * resNotSummed   # TODO here add how close to go & also try to even more force far clusters not to affect
            # #print(resNotAvg.shape)
            # res = resNotAvg.mean((1,))
            #print(res.shape)
            #res = torch.matmul(res, centers)
            #dists = dists.view(*dists.shape[1:])
            #print("!!!!", dists.shape)
            # print(dists.shape)
            closest = dists.argmin(-1)
            # print(points.shape, closest.shape, centers[closest].view(N, -1).shape)
            diffs = centers[closest].view(B, N, -1) - points
            res = pushDeg * diffs + points
            # print(diffs.shape)
        else:
            assert False

        # print(distsRev3dim.shape, dists3dim.shape, distsDiv.shape, powed.shape, denomin.shape, res.shape)
        #fcmNoPower = distsBy1 / distsBy1.sum(1).view(-1,1)  # it uses square distances, so it is like for m=2 NOT REALLY XD
        # print(dists.shape, fcmNoPower.shape, k, distsBy1.sum(1).view(-1,1).shape)
        # #fcm = torch.square(fcmNoPower)
        # print(dists, distsBy1, distsBy1.sum(1), fcmNoPower)
        return res


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
