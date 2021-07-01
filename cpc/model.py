# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch

import random
from PIL import Image, ImageDraw

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


class Smartpool(nn.Module):
    def __init__(
        self,
        factor,
        temperature=1e-5,
        in_channels=512,
        dim_mlp=2048,
        use_differences=False,
        smartaveraging_hardcoded_weights=False,
        smartaveraging_window_size=None,
        smartaveraging_loss_parameter=False,
        smartaveraging_hardcoded_window_size=None,
        entire_batch=False
    ):
        """Smart pooling algorithm

        Args:
            factor: factor by which the sequence's length will be reduced
            temperature: added when normalizing
        """
        super().__init__()

        self.factor = factor
        self.temperature = temperature
        self.entire_batch = entire_batch
        self.register_buffer("filters", torch.FloatTensor([[[[-1,1],[1,-1]]]]), persistent=False)
        self.in_channels = in_channels
        self.dim_mlp = dim_mlp
        self.use_differences = use_differences
        self.smartaveraging_hardcoded_weights = smartaveraging_hardcoded_weights
        self.smartaveraging_window_size = smartaveraging_window_size
        self.smartaveraging_loss_parameter = smartaveraging_loss_parameter
        self.smartaveraging_hardcoded_window_size = smartaveraging_hardcoded_window_size

        if self.smartaveraging_hardcoded_weights:
            #self.register_buffer("hardcoded_windows", torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            #                                                        [0.0, 0.1, 0.2, 0.4, 0.2, 0.1, 0.0], 
            #                                                        [0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0], 
            #                                                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))

            #self.register_buffer("hardcoded_windows", torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1 ,1 ,1,1,1,1,1,1,1,1, 1], 
            #                                                        [0.0, 0.0, 1,1,1,2,2,2,2,2, 2,2,2,2,1, 1,1, 0.0, 0.0], 
            #                                                        [0.0, 0.0, 0.0, 0,0,1,1,2,2,3,2, 2,1,1,0, 0,0.0, 0.0, 0.0],
            #                                                        [0.0, 0.0, 0.0, 0,0,0,1,1,2,3, 2,1,1,0,0, 0,0.0, 0.0, 0.0],
            #                                                        [0.0, 0.0, 0.0, 0,0,0,0,1,2,4, 2,1,0,0,0, 0,0.0, 0.0, 0.0],
            #                                                        [0.0, 0.0, 0.0, 0,0,0,0,0,2,5, 2,0,0,0,0, 0,0.0, 0.0, 0.0],
            #                                                        [0.0, 0.0, 0.0, 0,0,0,0,0,0,1, 0,0,0,0,0, 0,0.0, 0.0, 0.0]]))

            #self.register_buffer("hardcoded_windows", torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1 ,1 ,1,1,1,1,1,1,1,1, 1], 
            #                                                        [0.0, 0.0, 1,1,1,2,2,2,2,2, 2,2,2,2,1, 1,1, 0.0, 0.0], 
            #                                                        [0.0, 0.0, 0.0, 0,0,1,1,2,2,3,2, 2,1,1,0, 0,0.0, 0.0, 0.0]]))

            #self.register_buffer("hardcoded_windows", torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1 ,1 ,1,1,1,1,1,1,1,1, 1]]))

            #self.register_buffer("hardcoded_windows", torch.ones(1, 31))

            #self.register_buffer("hardcoded_windows", torch.ones(1, 51))

            self.register_buffer("hardcoded_windows", torch.ones(1, self.smartaveraging_hardcoded_window_size))

            #self.register_buffer("hardcoded_windows", torch.ones(1, 101))
            
            #self.register_buffer("hardcoded_windows", torch.ones(1, 127))

            self.hardcoded_windows /= self.hardcoded_windows.sum(1, keepdim=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.dim_mlp),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(self.dim_mlp, self.dim_mlp),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(self.dim_mlp, 1),
            nn.Sigmoid() if not self.smartaveraging_hardcoded_weights else nn.Sequential(nn.Linear(1, self.hardcoded_windows.shape[0]), nn.LogSoftmax(dim=-1))) if not self.use_differences else None

        #if self.mlp is not None:
        #    def inspect_mlp_grad(self, grad_input, grad_output):
        #            print('Inside ' + self.__class__.__name__ + ' backward')
        #            print('Inside class:' + self.__class__.__name__)
        #            print('')
        #            print('grad_input: ', type(grad_input))
        #            print('grad_input[0]: ', type(grad_input[0]))
        #            print('grad_output: ', type(grad_output))
        #            print('grad_output[0]: ', type(grad_output[0]))
        #            print('')
        #            print('grad_input size:', grad_input[0].size())
        #            print('grad_output size:', grad_output[0].size())
        #            print('grad_input norm:', grad_input[0].norm())
        #                        
        #    self.mlp.register_backward_hook(inspect_mlp_grad)
            
        self.visualization = False

    def warp(self, X, importance):
        importance_cs = importance.cumsum(1)

        # This really searches for the low boundary of each new pixel
        pixel_contributions = importance_cs.view(importance_cs.shape[0], -1, 1) - torch.arange(torch.round(importance_cs[0, -1]).item(), device=X.device).view(1, 1, -1)
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

    def smartaveraging(self, X, importance):
        B, T, C = X.shape
        window_size = self.smartaveraging_window_size if not self.smartaveraging_hardcoded_weights else self.hardcoded_windows.shape[1]
        assert window_size % 2 == 1

        if self.smartaveraging_hardcoded_weights:
            window_size = self.hardcoded_windows.shape[1]
            #weight_index = importance.argmax(2)
            weight_index = torch.zeros(B,T, device=X.device).long()
            values = self.hardcoded_windows[weight_index, :].view(-1)
        else:
            sigmas = torch.pow(10, 1 - 2 * importance.view(-1)) # From 1e1 to 1e-1
            windows = torch.arange(-(window_size // 2), window_size // 2 + 1, device=X.device).view(-1,1).repeat(1, sigmas.shape[0]) # window_size x (B x T)
            normal = torch.distributions.normal.Normal(0, sigmas)
            probs = normal.log_prob(windows).exp().T.view(B, T, window_size)
            probs = probs / probs.sum(dim=2, keepdim=True)
            values = probs.reshape(-1)

        X_padded = F.pad(X.unsqueeze(1), (0, 0, window_size // 2, window_size // 2), mode='reflect').squeeze(1)
        matrix = torch.zeros(B, T, T + window_size - 1, device=X.device)
        batch_index = torch.arange(B, device=X.device).repeat_interleave(T * window_size)
        time_index = torch.arange(0, T, device=X.device).repeat_interleave(window_size).repeat(B)
        window_index = (torch.arange(T, device=X.device).view(-1,1) + torch.arange(window_size, device=X.device).view(1,-1)).view(-1).repeat(B)
        matrix[batch_index, time_index, window_index] = values

        X_new = matrix @ X_padded
        return X_new

    #def forward(self, features):
    def forward(self, features, factor):
        B,T,C = features.size()
        self.factor = factor

        padding_mask = torch.zeros(B,T, dtype=torch.bool, device=features.device)
        padding_per_batch = (padding_mask > 0).sum(1)
        total_T = padding_mask.numel() - padding_per_batch.sum() if self.entire_batch else (T - padding_per_batch).view(-1, 1)
        
        if not self.use_differences:
            if self.smartaveraging_window_size is not None:
                importance = self.mlp(features).view(B,T)
            elif self.smartaveraging_hardcoded_weights:
                importance = self.mlp(features)
                importance = importance.view(B,T,-1)
            else:
                importance = self.mlp(features).view(B,T) + self.temperature
                importance = importance / importance.sum(1, keepdim=True) * (total_T / self.factor) # Reducing the original length T by some factor
        else:
            #features_tmp = F.pad(features, (0,0,1,0), value=features.mean().item())
            features_tmp = F.pad(features.unsqueeze(0), (0,0,1,0), mode='reflect').squeeze(0)
            importance = (features_tmp[:,1:,:] - features_tmp[:,:-1,:]).abs().sum(dim=2) + self.temperature
            importance = importance / importance.sum(1, keepdim=True) * (total_T / self.factor) # Reducing the original length T by some factor
        
        if self.visualization:
            return importance
       
        if self.smartaveraging_window_size is not None or self.smartaveraging_hardcoded_weights:
            features = self.smartaveraging(features, importance)
        else:
            features, _ = self.warp(features, importance)

        if self.smartaveraging_loss_parameter:
            return features, importance

        return features
    
    def set_visualization(self, value):
        self.visualization = value

class CPCSmartpoolEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm",
                 smartpoolingLayer=4,
                 noPadding=False,
                 dimMlp=2048,
                 useDifferences=False,
                 temperature=1e-5,
                 smartaveragingHardcodedWeights=False,
                 smartaveragingWindowSize=7,
                 smartaveragingLossParameter=False,
                 smartaveragingHardcodedWindowSize=None):

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
        
        self.smartaveragingLossParameter = smartaveragingLossParameter
        self.smartpoolingLayer = smartpoolingLayer
        if smartpoolingLayer < 3 or smartpoolingLayer > 5:
            raise ValueError(f"SmartpoolingLayer must be between 3 and 5")

        self.paddings = [0, 0, 0, 0, 0] if noPadding else [3, 2, 1, 1, 1]
        self.temperature = temperature

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=self.paddings[0], padding_mode='reflect')
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=self.paddings[1], padding_mode='reflect')
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=self.paddings[2], padding_mode='reflect')
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = Smartpool(4, temperature=temperature, in_channels=sizeHidden, dim_mlp=dimMlp, use_differences=useDifferences) if smartpoolingLayer == 3 else nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=self.paddings[3], padding_mode='reflect')
        self.batchNorm3 = normLayer(sizeHidden)
        if smartpoolingLayer >= 4:
            self.conv4 = Smartpool(2, temperature=temperature, in_channels=sizeHidden, dim_mlp=dimMlp, use_differences=useDifferences) if smartpoolingLayer == 4 else nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=self.paddings[4], padding_mode='reflect')
            self.batchNorm4 = normLayer(sizeHidden)
        if smartpoolingLayer == 5:
            self.conv5 = Smartpool(1, temperature=temperature, in_channels=sizeHidden, dim_mlp=dimMlp, use_differences=useDifferences, smartaveraging_hardcoded_weights=smartaveragingHardcodedWeights, smartaveraging_window_size=smartaveragingWindowSize, smartaveraging_loss_parameter=smartaveragingLossParameter, smartaveraging_hardcoded_window_size=smartaveragingHardcodedWindowSize)
            self.batchNorm5 = normLayer(sizeHidden)

        self.DOWNSAMPLING = 160
        

    def getDimOutput(self):
        return self.dimEncoded


    def forward(self, x):
        #def start_debugger():
        #    import ptvsd
        #    ptvsd.enable_attach(('0.0.0.0', 7310))
        #    print("Attach debugger now")
        #    ptvsd.wait_for_attach()
        #    print(f"Starting debugger")

        T_out = x.shape[2] // self.DOWNSAMPLING
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x))) 
        x = F.relu(self.batchNorm2(self.conv2(x)))

        factor = x.shape[2] / T_out
        x = self.conv3(x.transpose(1,2), factor).transpose(1,2) if self.smartpoolingLayer == 3 else self.conv3(x)
        x = F.relu(self.batchNorm3(x))

        if self.smartpoolingLayer >= 4: 
            factor = x.shape[2] / T_out
            x = self.conv4(x.transpose(1,2), factor).transpose(1,2) if self.smartpoolingLayer == 4 else self.conv4(x)
            x = F.relu(self.batchNorm4(x))

        if self.smartpoolingLayer == 5:
            factor = x.shape[2] / T_out
            
            if self.smartaveragingLossParameter:
                x, importance = self.conv5(x.transpose(1,2), factor)
                x = x.transpose(1,2)
                x = F.relu(self.batchNorm5(x))
                return x, importance
        
            x = self.conv5(x.transpose(1,2), factor).transpose(1,2)
            x = F.relu(self.batchNorm5(x))

        return x
    
    def visualize(self, x):
        #def start_debugger():
        #    import ptvsd
        #    ptvsd.enable_attach(('0.0.0.0', 7310))
        #    print("Attach debugger now")
        #    ptvsd.wait_for_attach()
        #    print(f"Starting debugger")

        if self.smartpoolingLayer == 3:
            self.conv3.set_visualization(True)
        elif self.smartpoolingLayer == 4:
            self.conv4.set_visualization(True)
        else:
            self.conv5.set_visualization(True)

        T_out = x.shape[2] // self.DOWNSAMPLING
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x))) 
        x = F.relu(self.batchNorm2(self.conv2(x)))

        factor = x.shape[2] / T_out
        x = self.conv3(x.transpose(1,2), factor) if self.smartpoolingLayer == 3 else self.conv3(x)
        if self.smartpoolingLayer == 3:
            self.conv3.set_visualization(False)
            return x
        x = F.relu(self.batchNorm3(x))

        if self.smartpoolingLayer >= 4: 
            factor = x.shape[2] / T_out
            x = self.conv4(x.transpose(1,2), factor) if self.smartpoolingLayer == 4 else self.conv4(x)
            if self.smartpoolingLayer == 4:
                self.conv4.set_visualization(False)
                return x
            x = F.relu(self.batchNorm4(x))

        if self.smartpoolingLayer == 5:
            factor = x.shape[2] / T_out
            x = self.conv5(x.transpose(1,2), factor)
            self.conv5.set_visualization(False)
            return x
            x = F.relu(self.batchNorm5(x))

        return x
        

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
                 reverse=False,
                 smartpoolingConfig=None):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        self.is_conv5_frozen = False
        self.conv5 = None
        self.batchNorm5 = None
        self.smartaveraging_loss_parameter = False

        if smartpoolingConfig is not None:
            dimMlp, useDifferences, temperature, smartaveragingHardcodedWeights, smartaveragingWindowSize, smartaveragingLossParameter, smartaveragingHardcodedWindowSize = smartpoolingConfig
            self.conv5 = Smartpool(1, temperature=temperature, in_channels=dimEncoded, dim_mlp=dimMlp, use_differences=useDifferences, smartaveraging_hardcoded_weights=smartaveragingHardcodedWeights, smartaveraging_window_size=smartaveragingWindowSize, smartaveraging_loss_parameter=smartaveragingLossParameter, smartaveraging_hardcoded_window_size=smartaveragingHardcodedWindowSize)
            self.batchNorm5 = ChannelNorm(dimEncoded)
            self.smartaveraging_loss_parameter = smartaveragingLossParameter

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

    def visualize(self, x):
        factor = 1
        self.conv5.set_visualization(True)
        importance = self.conv5(x, factor)
        self.conv5.set_visualization(False)
        return importance

    def forward(self, x):
        if self.conv5 is not None and not self.is_conv5_frozen:
            factor = 1
            if self.smartaveraging_loss_parameter:
                x, importance = self.conv5(x, factor)
            else:
                x = self.conv5(x, factor)
            x = F.relu(self.batchNorm5(x.transpose(1,2)).transpose(1,2))

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

        if self.smartaveraging_loss_parameter and not self.is_conv5_frozen:
            return x, importance

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
                 AR,
                 smartpoolingInAR=False,
                 smartaveragingLossParameter=False):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.smartpoolingInAR=smartpoolingInAR
        self.smartaveragingLossParameter = smartaveragingLossParameter

    def forward(self, batchData, label):
        if self.smartaveragingLossParameter and not (self.smartpoolingInAR and self.gAR.is_conv5_frozen):
            if self.smartpoolingInAR:
                encodedData = self.gEncoder(batchData).permute(0, 2, 1)
                cFeature, importance = self.gAR(encodedData)
            else:
                encodedData, importance = self.gEncoder(batchData)
                encodedData = encodedData.permute(0, 2, 1)
                cFeature = self.gAR(encodedData)

            return cFeature, encodedData, label, importance

        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label

    def disableSmartaveragingLossParameter(self):
        if self.smartaveragingLossParameter:
            self.smartaveragingLossParameter = False
            if self.smartpoolingInAR:
                self.gAR.smartaveraging_loss_parameter = False
                self.gAR.conv5.smartaveraging_loss_parameter = False
            else:
                self.gEncoder.smartaveragingLossParameter = False
                self.gEncoder.conv5.smartaveraging_loss_parameter = False

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
