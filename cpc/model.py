# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch

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

class Smartpool(nn.Module):
    def __init__(
        self,
        factor,
        search_perc,
        in_channels=512,
        entire_batch=False,
        mlp2=False
    ):
        """Smart pooling algorithm

        Args:
            factor: factor by which the sequence's length will be reduced
            search_perc: percentage of length of sequence after smartpooling to search for border. Ideally the border is located somewhere in +-search_perc
        """
        super().__init__()

        self.search_perc = search_perc
        self.factor = factor
        self.entire_batch = entire_batch
        self.register_buffer("filters", torch.FloatTensor([[[[-1,1],[1,-1]]]]), persistent=False)
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, 2048),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(2048, 1),
            nn.Sigmoid())
        
        if mlp2 == True:
            self.mlp2 = nn.Sequential(
                nn.Linear(2, 256),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(256,512),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(512,256),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.Linear(256,1))
        else:
            self.mlp2 = None
            
        self.visualization = False

    def warp(self, X, new_lens):
        new_lens_cs = new_lens.cumsum(1)
        #print(f"new_lens_cs (shape {new_lens_cs.shape}) last elements: {new_lens_cs[:,-1]}")
        # This really searches for the low boundary of each new pixel
        try:
            pixel_contributions = new_lens_cs.view(new_lens_cs.shape[0], -1, 1) - torch.arange(torch.round(new_lens_cs[0, -1]).item(), device=X.device).view(1, 1, -1)
        except:
            print(f"new_lens_cs (shape {new_lens_cs.shape}) last elements: {new_lens_cs[:,-1]}")
        pixel_contributions = pixel_contributions.view(X.size(0), X.size(1), pixel_contributions.size(2))
        # Zero out the negative contributions, i.e. pixels which come before each row                              
        pixel_contributions = torch.max(torch.tensor(0.0, device=X.device), pixel_contributions)       

        # # This contains the cumulated pixel lengths for all pixels in each 
        # pixel_contributions
    
        pixel_contributions = pixel_contributions.unsqueeze(1)
        interp_weights = F.conv2d(pixel_contributions, self.filters, padding=1)
        interp_weights = interp_weights[:,:,:-1,1:] # Removing padding
        interp_weights = interp_weights.squeeze(1)

        # # Each column corresponds to a new element. Its values are the 
        # # weights associated with the original data.
        # interp_weights

        interp_weights = interp_weights.transpose(1, 2)
        Xnew = interp_weights @ X
        return Xnew, interp_weights

    def nonzero_interval_length(self, x, dim):
        nonz = (x > 0)
        _, low = ((nonz.cumsum(dim) == 1) & nonz).max(dim, keepdim=True)
        rev_cumsum = nonz.long().flip(dim).cumsum(dim).flip(dim)
        _, high = ((rev_cumsum == 1) & nonz).max(dim, keepdim=True)
        
        return high - low + 1

    def forward(self, features):
        B,T,C = features.size()

        padding_mask = torch.zeros(B,T, dtype=torch.bool, device=features.device)
        padding_per_batch = (padding_mask > 0).sum(1)
        total_T = padding_mask.numel() - padding_per_batch.sum() if self.entire_batch else (T - padding_per_batch).view(-1, 1)
        
        new_lens = self.mlp(features).view(B,T)
        new_lens = new_lens / new_lens.sum(1, keepdim=True) * (total_T / self.factor) # Reducing the original length T by some factor 
        
        if self.visualization:
            return new_lens
       
        features, interp_weights = self.warp(features, new_lens)
        
        if self.mlp2 is not None:
            features = self.mlp2(features)

        return features
    
    def set_visualization(self, value):
        self.visualization = value

class DoXTimes(nn.Module):
    def __init__(self, model, classifier, features=None):
        super().__init__()
        self.model = model
        self.classifier = classifier
        self.features = features
        
    def forward(self, x):
        #print('1', x.shape)
        
        #print('2', x.shape)
        if self.features is not None:

            x = x.unsqueeze(1)
            x = self.features(x)
            x = x.squeeze(2)
                
        #print('3', x.shape)
        x = x.transpose(1,2)
        B = x.shape[0]
        #x = torch.cat([self.model(x[i].unsqueeze(0)) for i in range(B)])
        x = self.model(x)
        #print('4', x.shape)
        x = self.classifier(x)
        x = x.view(B * x.shape[1], -1)
        return x
    
    def visualize(self, x):
        self.model.set_visualization(True)
        if self.features is not None:
            x = x.unsqueeze(1)
            x = self.features(x)
            x = x.squeeze(2)
                
        #print('3', x.shape)
        x = x.transpose(1,2)
        B = x.shape[0]
        x = torch.cat([self.model(x[i].unsqueeze(0)) for i in range(B)])
        
        x = x.squeeze(1)
        self.model.set_visualization(False)
        return x

class CPCSmartpoolEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm"):

        super(CPCSmartpoolEncoder, self).__init__()

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
        #self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.conv4 = Smartpool(2, 0.3, sizeHidden)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv3.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = self.conv4(x.transpose(1,2)).transpose(1,2)
        x = F.relu(self.batchNorm4(x))
        #x = F.relu(self.batchNorm4(self.conv4(x)))
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


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label

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
