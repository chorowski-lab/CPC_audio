

import torch
import time
from heapq import *

def computeSEcosts(enc, maxSegmLen):

    # enc: B x N x D

    linSums = torch.zeros((maxSegmLen, *(enc.shape)), dtype=torch.float32)
    sqSums = torch.zeros((maxSegmLen, *(enc.shape)), dtype=torch.float32)

    encSq = torch.clamp(torch.square(enc), min=0)
    linSums[0] = enc
    sqSums[0] = encSq

    for i in range(1,maxSegmLen):
        linSums[i, :, 1:, :] = linSums[i-1, :, :-1, :] + enc[:, 1:, :]
        sqSums[i, :, 1:, :] = sqSums[i-1, :, :-1, :] + encSq[:, 1:, :]

    costs = torch.zeros((maxSegmLen, *(enc.shape[:-1])), dtype=torch.float32)

    for i in range(maxSegmLen):
        costs[i] = (sqSums[i] - torch.square(linSums[i]) / (i+1)).sum(-1)

    return costs



def costSegm(costs, shape, k, minSegmsInLine):
    maxSegmLen = costs.shape[0]
    print(f"maxsegmlen: {maxSegmLen}")
    # shape: B x N
    h, w = shape[0], shape[1]
    print("!!", h, w)
    segms = set()  #{}
    lenOnRight = {}
    lenOnLeft = {}
    for i in range(h):
        for j in range(w):
            #print(i,j)
            segms.add((i,j,1)) #= costs[0,i,j]
            lenOnRight[(i,j)] = 1
            lenOnLeft[(i,j)] = 1
    pq = []
    for i in range(h):
        for j in range(w-1):
            heappush(pq, (costs[1,i,j+1].item(),i,j,1,i,j+1,1))
    linesSegms = [w for _ in range(h)]
    #numSegms = len(segms)
    loopIters = 0
    print("--", len(pq))
    while len(pq) > 0 and len(segms) > k:  #numSegms > k:
        loopIters += 1
        cost, i1, j1, l1, i2, j2, l2 = heappop(pq)
        if (i1,j1,l1) not in segms or (i2,j2,l2) not in segms or linesSegms[i1] <= minSegmsInLine:
            continue
        #print(":", i1, j1, l1, i2, j2, l2)
        
        segms.remove((i1,j1,l1))
        segms.remove((i2,j2,l2))
        newLen = l1+l2
        segms.add((i1, j2, l1+l2))  # = costs[l1+l2-1, i1, j2]
        linesSegms[i1] -= 1
        lenOnRight[(i1,j1-l1+1)] = newLen
        lenOnLeft[(i1,j2)] = newLen
        #numSegms -= 1
        #print("@", newLen, i1, j2-newLen+1, j2, "|", j1-l1+1, j2)
        if j1-l1 >= 0:
            ll = lenOnLeft[(i1,j1-l1)]
            #print("ll", ll)
            if newLen+ll <= maxSegmLen:
                heappush(pq, (costs[newLen+ll-1,i,j2].item(),i1,j1-l1,ll,i1,j2,newLen))
                #print((i1,j1-l1,ll) in segms, (i1,j2,newLen) in segms)
        if j2+1 < w:
            lr = lenOnRight[(i1, j2+1)]
            #print("lr", lr)
            if newLen+lr <= maxSegmLen:
                heappush(pq, (costs[newLen+lr-1,i,j2+lr].item(),i1,j2,newLen,i1,j2+lr,lr))
                #print((i1,j2,newLen) in segms, (i1,j2+lr,lr) in segms)
        #print(len(pq), len(segms))
        #print(segms)
    print(len(pq), len(segms))

    print(f"Loop iters: {loopIters}")
    return segms






enc1 = torch.tensor([[[1.],[1.],[2.],[3.]], [[4.],[5.],[6.],[7.]]])
costs1 = computeSEcosts(enc1, 4)
segms1 = costSegm(costs1, enc1.shape, 5, 2)
print(costs1)
print(segms1)
print("----------")
enc2 = torch.tensor([[[1.,1.],[1.,1.],[2.,2.],[3.,3.]], [[4.,4.],[5.,5.],[6.,6.],[7.,7.]]])
print(computeSEcosts(enc2, 4))


enc3 = torch.rand(64, 128, 256, dtype=torch.float32)
t0 = time.time()
costs = computeSEcosts(enc3, 10)
#print("***", costs.shape)
t1 = time.time()
costs=costs.cpu()
t2 = time.time()
segms = costSegm(costs, enc3.shape, 2000, 2)
t3 = time.time()
print(len(segms))
print(f"Normal batch time: {t1-t0}, {t2-t1}, {t3-t2} | sum: {t3-t0}")

t0 = time.time()
d = {}
for i in range(1000000):
    d[i] = i
t1 = time.time()
print(f"sth time: {t1-t0}")

# print("::::::::::::::")

# enc3 = torch.rand(2, 25, 256, dtype=torch.float32)
# t0 = time.time()
# costs = computeSEcosts(enc3, 10)
# #print("***", costs.shape)
# t1 = time.time()
# costs=costs.cpu()
# t2 = time.time()
# segms = costSegm(costs, enc3.shape, 10, 2)
# t3 = time.time()
# print(len(segms))
# print(f"Normal batch time: {t1-t0}, {t2-t1}, {t3-t2}")