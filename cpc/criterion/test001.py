

import torch
from cpc.criterion.criterion import TimeAlignedPredictionNetwork

pred = TimeAlignedPredictionNetwork(3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2],[5,0.2],[6,0.2],[7,0.2]],
                        [[1,0.2],[2,0.2],[3,0.2],[4,0.2],[5,0.2],[6,0.2],[7,0.2]]]).cuda()
candidates = torch.zeros_like(featc).view(1,featc.shape[0],1,featc.shape[1],featc.shape[2]).repeat(3,1,5,1,1).cuda()

print(featc.shape, candidates.shape)

print(featc[:,:,-1])

pred(featc[:,:-3,:], candidates[:,:,:,:-3,:], featc[:,:,-1])