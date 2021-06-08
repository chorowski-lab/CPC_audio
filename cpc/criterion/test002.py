

import torch
from cpc.criterion.soft_align import TimeAlignedPredictionNetwork

# predStartDep  simple
pred = TimeAlignedPredictionNetwork(2, 3, 3, rnnMode='LSTM', dropout=False, sizeInputSeq=5, mode="predStartDep",
    teachOnlyLastFrameLength=True).cuda()

featc = torch.tensor([[[1,0.2,0.2],[1,0.2,0.2],[1,0.2,0.2],[1,0.2,0.2],[1,0.2,0.2],[1,0.7,0.2],[1,0.7,0.7]],
                        [[1,0.2,0.2],[1,0.2,0.2],[1,0.2,0.3],[1,0.2,0.4],[1,0.2,0.5],[1,0.2,0.2],[1,0.2,0.2]]]).cuda()
#candidates = torch.zeros_like(featc).view(1,featc.shape[0],1,featc.shape[1],featc.shape[2]).repeat(3,1,5,1,1).cuda()

#print(featc.shape, candidates.shape)

print(featc[:,:,-1])

pred(featc[:,:-2,:], featc[:,:,-2:])  # featc[:,:,-2:])  featc[:,:,-1])