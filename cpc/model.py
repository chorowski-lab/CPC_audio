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
                 normMode="layerNorm",
                 paddingMode="zeros"):

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
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3, padding_mode=paddingMode)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2, padding_mode=paddingMode)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1, padding_mode=paddingMode)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1, padding_mode=paddingMode)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1, padding_mode=paddingMode)
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


def sequence_segmenter(encodedData, final_lengt_factor, step_reduction=0.2):
    assert not torch.isnan(encodedData).any()
    device = encodedData.device
    encFlat = F.pad(encodedData.reshape(-1, encodedData.size(-1)).detach(), (0, 0, 1, 0))
    feat_csum = encFlat.cumsum(0)
    feat_csum2 = (encFlat**2).cumsum(0)
    idx = torch.arange(feat_csum.size(0), device=feat_csum.device)

    final_length = int(final_lengt_factor * len(encFlat))

    while len(idx) > final_length:
        begs = idx[:-2]
        ends = idx[2:]

        sum1 = (feat_csum.index_select(0, ends) - feat_csum.index_select(0, begs))
        sum2 = (feat_csum2.index_select(0, ends) - feat_csum2.index_select(0, begs))
        num_elem = (ends-begs).float().unsqueeze(1)

        diffs = F.pad(torch.sqrt(((sum2/ num_elem - (sum1/ num_elem)**2) ).mean(1)), (1,1), value=1e10)

        num_to_retain = max(final_length, int(idx.shape[-1] * step_reduction))
        _, keep_idx = torch.topk(diffs, num_to_retain)
        keep_idx = torch.sort(keep_idx)[0]
        idx = idx.index_select(0, keep_idx)
    
    # Ensure that minibatch boundaries are preserved
    seq_end_idx = torch.arange(0, encodedData.size(0)*encodedData.size(1), encodedData.size(1), device=device)
    idx = torch.unique(torch.cat((idx, seq_end_idx)), sorted=True)

    # now work out cut indices in each minibatch element
    batch_elem_idx = idx // encodedData.size(1)
    transition_idx = F.pad(torch.nonzero(batch_elem_idx[1:] != batch_elem_idx[:-1]), (0,0, 1,0))
    cutpoints = (torch.nonzero((idx % encodedData.size(1)) == 0))
    compressed_lens = (cutpoints[1:]-cutpoints[:-1]).squeeze(1)

    seq_idx = torch.nn.utils.rnn.pad_sequence(
        torch.split(idx[1:] % encodedData.size(1), tuple(cutpoints[1:]-cutpoints[:-1])), batch_first=True)
    seq_idx[seq_idx==0] = encodedData.size(1)
    seq_idx = F.pad(seq_idx, (1,0,0,0))

    frame_idxs = torch.arange(encodedData.size(1), device=device).view(1, 1, -1)
    compress_matrices = (
        (seq_idx[:,:-1, None] <= frame_idxs)
        & (seq_idx[:,1:, None] > frame_idxs)
    ).float()

    compressed_lens = compressed_lens.cpu()
    assert compress_matrices.shape[0] == encodedData.shape[0]
    return compress_matrices, compressed_lens


def compress_batch(encodedData, compress_matrices, compressed_lens, pack=False):
    ret = torch.bmm(
        compress_matrices / torch.maximum(compress_matrices.sum(-1, keepdim=True), torch.ones(1, device=compress_matrices.device)), 
        encodedData)
    if pack:
        ret = torch.nn.utils.rnn.pack_padded_sequence(
                ret, compressed_lens, batch_first=True, enforce_sorted=False
            )
    return ret


def decompress_padded_batch(compressed_data, compress_matrices, compressed_lens):
    if isinstance(compressed_data, torch.nn.utils.rnn.PackedSequence):
        compressed_data, unused_lens = torch.nn.utils.rnn.pad_packed_sequence(
            compressed_data, batch_first=True, total_length=compress_matrices.size(1))
    assert (compress_matrices.sum(1) == 1).all()
    return torch.bmm(
        compress_matrices.transpose(1, 2), compressed_data)


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False,
                 final_lengt_factor=None, 
                 step_reduction=0.2
                 ):

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
        self.final_lengt_factor = final_lengt_factor
        self.step_reduction = step_reduction

    def extra_repr(self):
        extras = [
            f'reverse={self.reverse}',
            f'final_lengt_factor={self.final_lengt_factor}',
            f'step_reduction={self.step_reduction}',
        ]
        return ', '.join(extras)


    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass

        if self.final_lengt_factor is None:
            x, h = self.baseNet(x, self.hidden)
        else:
            compress_matrices, compressed_lens = sequence_segmenter(
                x, self.final_lengt_factor, self.step_reduction)
            packed_compressed_x = compress_batch(
                x, compress_matrices, compressed_lens, pack=True
            )
            packed_x, packed_h = self.baseNet(packed_compressed_x)
            x = decompress_padded_batch(packed_x, compress_matrices, compressed_lens)

        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(h_elem.detach() for h_elem in h)
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
