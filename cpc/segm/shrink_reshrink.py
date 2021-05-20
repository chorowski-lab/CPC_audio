

import torch
from copy import deepcopy

def segmSetLengthChangeThings(segmSet, shape, D):
    # 2 things needed: indices to sum with _scatter_add, and numbers of things in rows
    srt = sorted(segmSet)
    # shape here is shape without last dim! : B x N
    B = shape[0]
    N = shape[1]
    fullLen = B*N
    scatterIndices = torch.fill_(torch.zeros(shape, dtype=torch.long).cuda(), -1).cpu()
    scatterLengths = torch.fill_(torch.zeros(shape, dtype=torch.long).cuda(), -1).cpu()
    # [!] segmSet MUST contain entry por every place inside shape, otherwise problems here
    # would then need to add in some additional place and then remove it
    
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
    return scatterIndices.cuda().view(-1,1).repeat(1,D), scatterLengths.cuda(), numsInLines, maxInLine


def shrinkSegms(batchLong, scatterIndices, maxInLine, lengths=None):

    # this is (lengths != None) for segmentation when segmSet and segmSetLengthChangeThings computed,
    # and also (lengths == None) for after-AR-length-restoring-layer's backward

    B = batchLong.shape[0]
    N = batchLong.shape[1]
    D = batchLong.shape[2]
    scatterLen = B*N  

    # batchLong already on cuda
    # actually those below also
    scatterIndices = scatterIndices.cuda()
    lengths = lengths.cuda()
    
    scatterTens = torch.zeros((scatterLen,D), dtype=torch.float32).cuda()
    #print("!", scatterTens.shape, scatterIndices.shape, batchLong.view(B*N, D).shape)
    #print(scatterIndices)
    #print("@", batchLong.shape, lengths.view(*(lengths.shape), -1).shape)
    if lengths is not None:  # segmentation forward, need to average things
        batchDivByLens = batchLong / torch.clamp(lengths.view(*(lengths.shape), -1), min=1.)
    else:  # restore-length-layer backward, need to sum gradients as shrinked things impact on all length
        batchDivByLens = batchLong
    scattered = scatterTens.scatter_add_(0, scatterIndices, batchDivByLens.view(B*N, D))
    return scattered.view(B,N,-1)[:, :maxInLine]


def expandSegms(batchShrinked, numsInLinesOrig, fullShape, lengthsForAveraging=None):

    B = batchShrinked.shape[0]
    Nshrinked = batchShrinked.shape[1]

    # batchShrinked already on cuda

    restored = torch.zeros(fullShape, dtype=torch.float32).cuda()
    numsInLines = deepcopy(numsInLinesOrig)
    #print(numsInLines)
    for line in range(len(numsInLines)):
        zerosToAdd = Nshrinked - len(numsInLines[line])
        numsInLines[line] = numsInLines[line] + [0 for _ in range(zerosToAdd)]  # concat
        #print("!", batchShrinked[line], torch.tensor(numsInLines[line]))
        restored[line] = torch.repeat_interleave(batchShrinked[line], torch.tensor(numsInLines[line].cuda()), dim=0)

    # this is the case when backward segmentation - want to pump each segm to length before segmentation,
    # but operating on dx, so need to make each place contirbution sum up to what was there
    # so need an average and not a copy
    # ; when this is None, the case is forward after AR to restore lengths
    if lengthsForAveraging is not None:
        restored = restored / torch.clamp(lengthsForAveraging.view(*(lengthsForAveraging.shape), -1), min=1.)

    return restored
        
if __name__ == "__main__":
    
    print([1,2]+[3,4])

    segms = set([(0,1,2), (1,0,1), (1,4,2), (0,4,3), (1,2,2)])
    #  00111
    #  23344
    print(segms)

    indices, lengths, numsInLines, maxInLine = segmSetLengthChangeThings(segms, (2,5), 3)
    print(indices, lengths, numsInLines, maxInLine)

    points = torch.tensor([[[1.,1.,1.], [2.,2.,2.], [3.,3.,5.], [4.,4.,5.], [5.,5.,5.]], 
                        [[1.,1.,1.], [2.,2.,2.], [3.,3.,3.], [4.,4.,4.], [5.,5.,5.]]])

    print(points.shape, indices.shape)
    shrinked = shrinkSegms(points, indices, maxInLine, lengths)
    print(shrinked)

    # TODO test it and on CUDA also
    shrinkedBackprop = shrinkSegms(points, indices, maxInLine)
    print(shrinkedBackprop)

    restored = expandSegms(shrinked, numsInLines, points.shape)
    print(restored)

    # TODO test it and on CUDA also
    restoredMean = expandSegms(shrinked, numsInLines, points.shape, lengths)
    print(restoredMean)