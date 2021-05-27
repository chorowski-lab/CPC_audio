

import torch
from copy import deepcopy
import sys
import time

def segmSetLengthChangeThings(segmSet, shape, D):
    # 2 things needed: indices to sum with _scatter_add, and numbers of things in rows
    #print("A")
    #sys.stdout.flush()
    srt = sorted(segmSet)
    #print("B")
    #sys.stdout.flush()
    # shape here is shape without last dim! : B x N
    B = shape[0]
    N = shape[1]
    fullLen = B*N
    scatterIndices = torch.fill_(torch.zeros(shape, dtype=torch.long).cpu(), -1).cpu().numpy()
    scatterLengths = torch.fill_(torch.zeros(shape, dtype=torch.int32).cpu(), -1).cpu().numpy()
    # [!] segmSet MUST contain entry por every place inside shape, otherwise problems here
    # would then need to add in some additional place and then remove it
    #print("C")
    #sys.stdout.flush()
    lastLine = -1
    startInThisLine = 0
    numsInLines = []
    for i, j, l in srt:
        startInThisLine += 1  # if new line, will be zeroed below
        if i != lastLine:
            numsInLines.append([])
            lastLine = i
            startInThisLine = 0
        numsInLines[-1].append(l)
        scatterIndices[i, j-l+1:j+1] = lastLine*N + startInThisLine
        scatterLengths[i, j-l+1:j+1] = l
    maxInLine = max(map(len, numsInLines))
    #print("D")
    #sys.stdout.flush()
    return torch.tensor(scatterIndices).view(-1,1).cuda().repeat(1,D), torch.tensor(scatterLengths).cuda(), numsInLines, maxInLine


def shrinkSegms(batchLong, scatterIndices, maxInLine, lengths=None):

    # this is (lengths != None) for segmentation when segmSet and segmSetLengthChangeThings computed,
    # and also (lengths == None) for after-AR-length-restoring-layer's backward
    #%#t0 = time.time()
    B = batchLong.shape[0]
    N = batchLong.shape[1]
    D = batchLong.shape[2]
    scatterLen = B*N  

    #%#t1 = time.time()
    # batchLong already on cuda
    # actually those below also
    batchLong = batchLong.cuda()
    scatterIndices = scatterIndices.cuda()
    if lengths is not None:
        lengths = lengths.cuda()
    
    #%#t2 = time.time()
    scatterTens = torch.zeros((scatterLen,1), dtype=torch.float32).cuda().repeat(1,D)
    #%#t3 = time.time()
    #print("!", scatterTens.shape, scatterIndices.shape, batchLong.view(B*N, D).shape)
    #print(scatterIndices)
    #print("@", batchLong.shape, lengths.view(*(lengths.shape), -1).shape)
    if lengths is not None:  # segmentation forward, need to average things
        batchDivByLens = batchLong / torch.clamp(lengths.view(*(lengths.shape), -1), min=1.)
    else:  # restore-length-layer backward, need to sum gradients as shrinked things impact on all length
        batchDivByLens = batchLong
    #print("@@", scatterTens.shape, scatterIndices.shape, batchDivByLens.shape, B, N, D)
    #%#t4 = time.time()
    scattered = scatterTens.scatter_add_(0, scatterIndices, batchDivByLens.contiguous().view(B*N, D))
    #%#t5 = time.time()
    #%#print(f"shrink time: {t1-t0}, {t2-t1}, {t3-t2}, {t4-t3}, {t5-t4}")
    return scattered.view(B,N,-1)[:, :maxInLine]


def expandSegms(batchShrinked, numsInLinesOrig, fullShape, lengthsForAveraging=None):

    #%#t01 = time.time()
    B = batchShrinked.shape[0]
    Nshrinked = batchShrinked.shape[1]

    # batchShrinked already on cuda
    batchShrinked = batchShrinked.cuda()
    if lengthsForAveraging is not None:
        lengthsForAveraging = lengthsForAveraging.cuda()

    restored = torch.zeros((*(fullShape[:-1]),1), dtype=torch.float32).cuda().repeat(1,1,fullShape[-1])
    numsInLines = deepcopy(numsInLinesOrig)
    #print(numsInLines)
    #%#t0 = time.time()
    for line in range(len(numsInLines)):
        zerosToAdd = Nshrinked - len(numsInLines[line])
        numsInLines[line] = numsInLines[line] + [0 for _ in range(zerosToAdd)]  # concat
    #%#t1 = time.time()
    for line in range(len(numsInLines)):
        #print(":", batchShrinked[line], torch.tensor(numsInLines[line]))
        restored[line] = torch.repeat_interleave(batchShrinked[line], torch.tensor(numsInLines[line]).cuda(), dim=0)
    #%#t2 = time.time()
    
    # this is the case when backward segmentation - want to pump each segm to length before segmentation,
    # but operating on dx, so need to make each place contirbution sum up to what was there
    # so need an average and not a copy
    # ; when this is None, the case is forward after AR to restore lengths
    if lengthsForAveraging is not None:
        restored = restored / torch.clamp(lengthsForAveraging.view(*(lengthsForAveraging.shape), -1), min=1.)
    #%#t02 = time.time()
    #%#print(f"expand part times: {t0-t01}, {t1-t0}, {t2-t1}, {t02-t2}")
    return restored
        
if __name__ == "__main__":
    
    enc0 = torch.tensor([1.]).cuda()  # first move to GPU takes >2s, so would spoil timed results
    
    print([1,2]+[3,4])

    segms = set([(0,1,2), (1,0,1), (1,4,2), (0,4,3), (1,2,2)])
    #  00111
    #  23344
    print(segms)

    indices, lengths, numsInLines, maxInLine = segmSetLengthChangeThings(segms, (2,5), 3)
    print(indices, lengths, numsInLines, maxInLine)
    #indices = indices.cuda()
    #lengths = lengths.cuda()
    print("---------change things end")

    points = torch.tensor([[[1.,1.,1.], [2.,2.,2.], [3.,3.,5.], [4.,4.,5.], [5.,5.,5.]], 
                        [[1.,1.,1.], [2.,2.,2.], [3.,3.,3.], [4.,4.,4.], [5.,5.,5.]]])

    print(points.shape, indices.shape)
    shrinked = shrinkSegms(points, indices, maxInLine, lengths)
    print(shrinked)
    print("---------shrinked end")

    shrinkedBackprop = shrinkSegms(points, indices, maxInLine)
    print(shrinkedBackprop)
    print("---------shrinkedBackprop end")

    restored = expandSegms(shrinked, numsInLines, points.shape)
    print(restored)
    print("---------restored end")

    restoredMean = expandSegms(shrinked, numsInLines, points.shape, lengths)
    print(restoredMean)
    print("---------restoredMean end")