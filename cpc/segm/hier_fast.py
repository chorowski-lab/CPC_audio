

import torch
from torch.autograd import Function, Variable
import time
from heapq import *
from cpc.segm.shrink_reshrink import *
from cpc.segm.fast_segm_torch_conversions import *
import numpy as np
from math import sqrt, ceil
from copy import deepcopy


def mergeSlowStats(segmSetTens, label, numPhones):
    with torch.no_grad():
        label = label.cpu().numpy()
        segmSet = convertTens3ValueSetBack(segmSetTens)
        #%#print("!mergeStats", segmSetTens.shape, label.shape, len(segmSet))
        
        merges = torch.zeros(numPhones, numPhones, dtype=torch.float32).cpu().numpy()
        counts = torch.zeros(numPhones, dtype=torch.float32).cpu().numpy()
        for line, idxInLine, l in segmSet:
            #line2, begin, end = segmDict[(line, idxInLine)]
            begin = idxInLine - l + 1
            end = idxInLine
            labelsThere = list(map(lambda x: x.item(), label[line, begin:(end+1)]))
            for x in labelsThere:
                counts[x] += 1
                for y in labelsThere:
                    merges[x,y] += 1
                merges[x,x] -= 1
        return merges, counts


def computeSEcosts(enc, maxSegmLen):

    # enc: B x N x D

    #maxSegmLen = maxSegmLenSqrt * maxSegmLenSqrt

    # maxSegmLenSqrt = int(ceil(sqrt(maxSegmLen)))
    # maxSegmLen = maxSegmLenSqrt*maxSegmLenSqrt
    # print("!!!!!!", maxSegmLen)

    linSums = torch.zeros((maxSegmLen, *(enc.shape)), dtype=torch.float32).cuda()
    sqSums = torch.zeros((maxSegmLen, *(enc.shape)), dtype=torch.float32).cuda()

    encSq = torch.clamp(torch.square(enc), min=0)
    linSums[0] = enc
    sqSums[0] = encSq

    for i in range(1,maxSegmLen):  #Sqrt):
        linSums[i, :, 1:, :] = linSums[i-1, :, :-1, :] + enc[:, 1:, :]
        sqSums[i, :, 1:, :] = sqSums[i-1, :, :-1, :] + encSq[:, 1:, :]

    # for i in range(1,maxSegmLenSqrt):
    #     linSums[i*maxSegmLenSqrt:(i+1)*maxSegmLenSqrt, :, maxSegmLenSqrt:, :] = \
    #         linSums[(i-1)*maxSegmLenSqrt:i*maxSegmLenSqrt, :, :-maxSegmLenSqrt, :] + linSums[maxSegmLenSqrt-1, :, :-maxSegmLenSqrt, :]
    #     sqSums[i*maxSegmLenSqrt:(i+1)*maxSegmLenSqrt, :, maxSegmLenSqrt:, :] = \
    #         sqSums[(i-1)*maxSegmLenSqrt:i*maxSegmLenSqrt, :, :-maxSegmLenSqrt, :] + sqSums[maxSegmLenSqrt-1, :, :-maxSegmLenSqrt, :]

    costs = torch.zeros((maxSegmLen, *(enc.shape[:-1])), dtype=torch.float32).cuda()

    for i in range(maxSegmLen):
        costs[i] = (sqSums[i] - torch.square(linSums[i]) / (i+1)).sum(-1)

    return costs


# maxCost has priority, but need to merge until later one from maxCost, k
# as need to report segmSet after maxCost and cost when k
# maxCost can be none and k can't
def costSegm(costs, shape, maxCost, k, minSegmsInLine):
    costs = costs.cpu().numpy()
    maxSegmLen = costs.shape[0]
    #print(f"maxsegmlen: {maxSegmLen}")
    # shape: B x N
    h, w = shape[0], shape[1]
    #print("!!", h, w)
    segms = set()  #{}
    lenOnRight = {}
    lenOnLeft = {}
    for iii in range(h):
        for jjj in range(w):
            #print(i,j)
            segms.add((iii,jjj,1)) #= costs[0,i,j]
            lenOnRight[(iii,jjj)] = 1
            lenOnLeft[(iii,jjj)] = 1
    pq = []
    for iii in range(h):
        for jjj in range(w-1):
            heappush(pq, (costs[1,iii,jjj+1].item(),iii,jjj,1,iii,jjj+1,1))
    linesSegms = [w for _ in range(h)]
    #numSegms = len(segms)
    loopIters = 0
    #print("--", len(pq))
    costNow = 0
    segmsWhenCost = None
    segmsWhenK = None
    costPresent = True
    if maxCost is None:
        maxCost = -1
        costPresent = False
    while len(pq) > 0 and (len(segms) > k or costNow <= maxCost):  #numSegms > k:
        loopIters += 1
        cost, i1, j1, l1, i2, j2, l2 = heappop(pq)
        if (i1,j1,l1) not in segms or (i2,j2,l2) not in segms or linesSegms[i1] <= minSegmsInLine:
            continue
        #print(":", i1, j1, l1, i2, j2, l2)
        
        ###print(";", cost)

        # before the merge that could have too big cost
        if costPresent and cost > maxCost and segmsWhenCost is None:
            ###print("A")
            segmsWhenCost = deepcopy(segms)
            costWhenCost = costNow
            kWhenCost = len(segms)

        costNow = cost

        segms.remove((i1,j1,l1))
        segms.remove((i2,j2,l2))
        newLen = l1+l2
        segms.add((i1, j2, l1+l2))  # = costs[l1+l2-1, i1, j2]
        linesSegms[i1] -= 1
        lenOnRight[(i1,j1-l1+1)] = newLen
        lenOnLeft[(i1,j2)] = newLen
        #numSegms -= 1
        ###print("@", newLen, i1, j2-newLen+1, j2, "|", j1-l1+1, j2, "&", i1, j1, l1, i2, j2, l2)
        if j1-l1 >= 0:
            ll = lenOnLeft[(i1,j1-l1)]
            ###print("ll", ll)
            if newLen+ll <= maxSegmLen:
                heappush(pq, (costs[newLen+ll-1,i1,j2].item(),i1,j1-l1,ll,i1,j2,newLen))
                ###print("ADD", newLen+ll-1, i1, j2, "|", (costs[newLen+ll-1,i1,j2].item(),i1,j1-l1,ll,i1,j2,newLen))
                #print((i1,j1-l1,ll) in segms, (i1,j2,newLen) in segms)
        if j2+1 < w:
            lr = lenOnRight[(i1, j2+1)]
            ###print("lr", lr)
            if newLen+lr <= maxSegmLen:
                heappush(pq, (costs[newLen+lr-1,i1,j2+lr].item(),i1,j2,newLen,i1,j2+lr,lr))
                ###print("ADD", newLen+lr-1,i1,j2+lr, "|", (costs[newLen+lr-1,i1,j2+lr].item(),i1,j2,newLen,i1,j2+lr,lr))
                #print((i1,j2,newLen) in segms, (i1,j2+lr,lr) in segms)
        #print(len(pq), len(segms))
        #print(segms)

        if len(segms) == k:
            ###print("B")
            segmsWhenK = deepcopy(segms)
            costWhenK = costNow
            kWhenK = k
        

    if costPresent and segmsWhenCost is None:
        ###print("C")
        segmsWhenCost = deepcopy(segms)
        costWhenCost = costNow
        kWhenCost = len(segms)

    # minsegmsinline reached
    if segmsWhenK is None:
        ###print("D")
        segmsWhenK = deepcopy(segms)
        costWhenK = costNow
        kWhenK = len(segms)

    #print(len(pq), len(segms))

    #print(f"Loop iters: {loopIters}")
    if costPresent:
        return (segmsWhenCost, costWhenCost, kWhenCost), (segmsWhenK, costWhenK, kWhenK)  #segms
    else:
        return None, (segmsWhenK, costWhenK, kWhenK)


class FastHierarchicalSegmentationLayer(Function):

    @staticmethod
    def getKforGivenShorteningAndShape(shape, shortening):
        numReprs = float(np.prod(shape[:-1]))
        return max(int(round(numReprs / float(shortening))), 1), int(numReprs)

    @staticmethod
    def forward(ctx, inputGPU, maxSegmentCost, k, maxSegmLen, minSegmsPerLine=None): 
        #print(f"input shape: {inputGPU.shape}")
        #%#print(f"FASTSEGM invoked, maxCost: {maxSegmentCost}, k : {k}")
        with torch.no_grad():
            #t0 = time.time()
            minSegmsPerLine = 5 if minSegmsPerLine is None else minSegmsPerLine
            seCosts = computeSEcosts(inputGPU, maxSegmLen)
            #t1 = time.time()
            segmDataCost, segmDataK = costSegm(seCosts.cpu(), inputGPU.shape, maxSegmentCost, k, minSegmsPerLine)
            segms, costForWantedK, _ = segmDataK
            costForWantedKTens = torch.tensor([costForWantedK]).cpu()
            ctx.mark_non_differentiable(costForWantedKTens)    
            if segmDataCost is None:
                actualK = k
            else:
                segms, _, actualK = segmDataCost
            actualKTens = torch.tensor([actualK]).cpu()
            ctx.mark_non_differentiable(actualKTens)
            #t2 = time.time()
            indices, lengths, numsInLines, maxInLine = segmSetLengthChangeThings(segms, (inputGPU.shape[0],inputGPU.shape[1]), inputGPU.shape[2])
            #t3 = time.time()
            #print(f"SHAPES: {indices.shape}, {lengths.shape}")
            shrinked = shrinkSegms(inputGPU, indices, maxInLine, lengths)
            #t4 = time.time()
            ctx.numsInLines = numsInLines
            ctx.lengths = lengths
            ctx.inputShape = inputGPU.shape
            ctx.mark_non_differentiable(lengths)
            ctx.mark_non_differentiable(indices)
            #print("!", convert2DimListsToInt32TensorAndMask(numsInLines))
            numInLinesConverted = convert2DimListsToInt32TensorAndMask(numsInLines)
            ctx.mark_non_differentiable(numInLinesConverted[0])
            ctx.mark_non_differentiable(numInLinesConverted[1])
            segmsConverted = convert3ValueIntSetToInt32Tens(segms)
            ctx.mark_non_differentiable(segmsConverted)
            maxNumConverted = convertNumToTens(maxInLine)
            ctx.mark_non_differentiable(maxNumConverted)
            shapeTens = torch.tensor(inputGPU.shape).cpu()
            ctx.mark_non_differentiable(shapeTens)
            
            #print(f"shape tens: {shapeTens}")
            #t5 = time.time()
            #print(f"actual time: {t1-t0}, {t2-t1}, {t3-t2}, {t4-t3}, nonsense conversion times: {t5-t4}")
            return shrinked, segmsConverted, indices, lengths, \
                numInLinesConverted[0], numInLinesConverted[1], maxNumConverted, shapeTens, costForWantedKTens, actualKTens

    @staticmethod
    def backward(ctx, dxThrough, segm=None, ind=None, lens=None, nums0=None, nums1=None, maxl=None, shapet=None, costt=None, actualkt=None):  #, finalSegments=None, segmentNumsInLines=None):

        with torch.no_grad():
            #print("!!!", dxThrough, ctx.numsInLines, ctx.lengths)
            backprop = expandSegms(dxThrough, ctx.numsInLines, ctx.inputShape, ctx.lengths)
            #print("***", backprop)
            return backprop, None, None, None, None, None


class FastSegmentationLengthRestoreLayer(Function):

    @staticmethod
    def forward(ctx, inputGPU, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, targetShape): # shrinkIndices & maxInLine only for backward
        #print(f"target shape: {targetShape}")
        numsInLines = convert2DimListTensBack((numsInLinesC0, numsInLinesC1))
        maxInLine = convertNumTensBack(maxInLine)
        with torch.no_grad():
            restored = expandSegms(inputGPU, numsInLines, targetShape)
            ctx.shrinkIndices = shrinkIndices
            ctx.maxInLine = maxInLine
            return restored

    @staticmethod
    def backward(ctx, dxThrough):  #, finalSegments=None, segmentNumsInLines=None):

        with torch.no_grad():
            #print("!!!-", dxThrough)
            backprop = shrinkSegms(dxThrough, ctx.shrinkIndices, ctx.maxInLine)
            #print("***-", backprop)
            return backprop, None, None, None, None, None



if __name__ == "__main__":

    enc0 = torch.tensor([1.]).cuda()  # first move to GPU takes >2s, so would spoil timed results

    enc1 = torch.tensor([[[1.],[1.],[2.],[3.]], [[4.],[5.],[6.],[7.]]]).cuda()
    costs1 = computeSEcosts(enc1, 4)
    segmsCost1, segmsK1 = costSegm(costs1, enc1.shape, None, 5, 2)
    segms1, _, _ = segmsK1
    print(costs1)
    print(segms1)
    print("*")
    print(segmsCost1)
    print(segmsK1)
    print("----")
    segmsCost2, segmsK2 = costSegm(costs1, enc1.shape, 0.3, 5, 2)
    segms2, _, _ = segmsK2
    #print(costs1)
    print(segms2)
    print("*")
    print(segmsCost2)
    print(segmsK2)
    print("----")
    segmsCost3, segmsK3 = costSegm(costs1, enc1.shape, 0.6, 3, 2)  # here can't merge because of minSegmsPerLine
    segms3, _, _ = segmsK3
    #print(costs1)
    print(segms3)
    print("*")
    print(segmsCost3)
    print(segmsK3)
    print("----")
    segmsCost4, segmsK4 = costSegm(costs1, enc1.shape, 0.6, 3, 1)
    segms4, _, _ = segmsK4
    #print(costs1)
    print(segms4)
    print("*")
    print(segmsCost4)
    print(segmsK4)
    print("----")
    segmsCost5, segmsK5 = costSegm(costs1, enc1.shape, 5, 3, 1)
    segms5, _, _ = segmsK5
    #print(costs1)
    print(segms5)
    print("*")
    print(segmsCost5)
    print(segmsK5)
    print("----")
    segmsCost6, segmsK6 = costSegm(costs1, enc1.shape, 5, 2, 1)
    segms6, _, _ = segmsK6
    #print(costs1)
    print(segms6)
    print("*")
    print(segmsCost6)
    print(segmsK6)
    print("----------")
    enc2 = torch.tensor([[[1.,1.],[1.,1.],[2.,2.],[3.,3.]], [[4.,4.],[5.,5.],[6.,6.],[7.,7.]]]).cuda()
    print(computeSEcosts(enc2, 4))


    enc3 = torch.rand(64, 128, 256, dtype=torch.float32).cuda()
    t0 = time.time()
    costs = computeSEcosts(enc3, 10)
    #print("***", costs.shape)
    t1 = time.time()
    costs=costs.cpu()
    t2 = time.time()
    segmsCost, segmsK = costSegm(costs, enc3.shape, None, 2000, 2)
    segms, _, _ = segmsK
    t3 = time.time()
    print(len(segms))
    t4_0 = time.time()
    indices, lengths, numsInLines, maxInLine = segmSetLengthChangeThings(segms, (enc3.shape[0],enc3.shape[1]), 256)
    #print("E")
    #sys.stdout.flush()
    t4 = time.time()
    indices = indices.cuda()
    lengths = lengths.cuda()
    print("!", indices.shape, lengths.shape)
    t5_0 = time.time()
    shrinked = shrinkSegms(enc3, indices, maxInLine, lengths)
    t5 = time.time()
    shrinkedBackprop = shrinkSegms(enc3, indices, maxInLine)
    t6 = time.time()
    restored = expandSegms(shrinked, numsInLines, enc3.shape)
    t7 = time.time()
    restoredMean = expandSegms(shrinked, numsInLines, enc3.shape, lengths)
    t8 = time.time()
    print(f"Normal batch time: {t1-t0}, {t2-t1}, {t3-t2} | sum: {t3-t0} | shrink things without grad: {t4-t4_0}, {t5-t5_0}, {t6-t5}, {t7-t6}, {t8-t7}")



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



    print("-------------------------- torch with shorten and restore ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float32)\
        .cuda().requires_grad_(True)
    
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeTens, costForKth, actualK = FastHierarchicalSegmentationLayer.apply(
        tensor, 
        None,
        5, 
        10,
        2)
    print(resOutput)
    print(convertTens3ValueSetBack(segms))
    print(shrinkIndices)
    print(lengths)
    print(convert2DimListTensBack((numsInLinesC0, numsInLinesC1)))
    print(convertNumTensBack(maxInLine))
    print(convertNumTensBack(costForKth))
    print(actualK)
    print("--")
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)
    tensor.grad.data.zero_()
    print("---")
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeTens))
    print(resOutRestored)
    resOutRestored.sum().backward()
    print(tensor.grad)  # 1s everywhere are correct, as sum is taken and stuff is copied length times after shortening

    print("-------------------------- torch with shorten and restore 2 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float32)\
        .cuda().requires_grad_(True)
    
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeTens, costForKth, actualK = FastHierarchicalSegmentationLayer.apply(
        tensor, 
        None,
        3, 
        10,
        1)
    print(resOutput)
    print(convertTens3ValueSetBack(segms))
    print(shrinkIndices)
    print(lengths)
    print(convert2DimListTensBack((numsInLinesC0, numsInLinesC1)))
    print(convertNumTensBack(maxInLine))
    print(convertNumTensBack(costForKth))
    print(actualK)
    print("--")
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)
    tensor.grad.data.zero_()
    print("---")
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeTens))
    print(resOutRestored)
    resOutRestored.sum().backward()
    print(tensor.grad)  # 1s everywhere are correct, as sum is taken and stuff is copied length times after shortening

    print("-------------------------- torch with shorten and restore 3 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float32)\
        .cuda().requires_grad_(True)
    
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeTens, costForKth, actualK = FastHierarchicalSegmentationLayer.apply(
        tensor, 
        20,
        3, 
        10,
        1)
    print(resOutput)
    print(convertTens3ValueSetBack(segms))
    print(shrinkIndices)
    print(lengths)
    print(convert2DimListTensBack((numsInLinesC0, numsInLinesC1)))
    print(convertNumTensBack(maxInLine))
    print(convertNumTensBack(costForKth))
    print(actualK)
    print("--")
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)
    tensor.grad.data.zero_()
    print("---")
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeTens))
    print(resOutRestored)
    resOutRestored.sum().backward()
    print(tensor.grad)  # 1s everywhere are correct, as sum is taken and stuff is copied length times after shortening

    print("-------------------------- torch with shorten and restore 4 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float32)\
        .cuda().requires_grad_(True)
    
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeTens, costForKth, actualK = FastHierarchicalSegmentationLayer.apply(
        tensor, 
        200,
        3, 
        10,
        1)  # this should only leave 2 segments as cost has priority over k
    print(resOutput)
    print(convertTens3ValueSetBack(segms))
    print(shrinkIndices)
    print(lengths)
    print(convert2DimListTensBack((numsInLinesC0, numsInLinesC1)))
    print(convertNumTensBack(maxInLine))
    print(convertNumTensBack(costForKth))
    print(actualK)
    print("--")
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)
    tensor.grad.data.zero_()
    print("---")
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeTens))
    print(resOutRestored)
    resOutRestored.sum().backward()
    print(tensor.grad)  # 1s everywhere are correct, as sum is taken and stuff is copied length times after shortening
    
    print("------------------measure batch time")
    encAllTimed = torch.rand(64, 128, 256, dtype=torch.float32).cuda().requires_grad_(True)
    ta0 = time.time()
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeEncAllTimedTens, _, _ = FastHierarchicalSegmentationLayer.apply(
        encAllTimed, 
        None,
        2000, 
        10,
        2)
    ta1 = time.time()
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeEncAllTimedTens))
    ta2 = time.time()
    resOutRestored.sum().backward()
    print(encAllTimed.grad.mean())
    ta3 = time.time()
    print(f"Measured time on 1 batch 64: forward ({ta1-ta0}, {ta2-ta1}), backward: ({ta3-ta2})")

    encAllTimed = torch.rand(32, 128, 256, dtype=torch.float32).cuda().requires_grad_(True)
    ta0 = time.time()
    resOutput, segms, shrinkIndices, lengths, numsInLinesC0, numsInLinesC1, maxInLine, inShapeEncAllTimedTens, _, _= FastHierarchicalSegmentationLayer.apply(
        encAllTimed, 
        None,
        1365, 
        10,
        2)
    ta1 = time.time()
    resOutRestored = FastSegmentationLengthRestoreLayer.apply(resOutput, numsInLinesC0, numsInLinesC1, shrinkIndices, maxInLine, torch.Size(inShapeEncAllTimedTens))
    ta2 = time.time()
    resOutRestored.sum().backward()
    print(encAllTimed.grad.mean())
    ta3 = time.time()
    print(f"Measured time on 1 batch 32: forward ({ta1-ta0}, {ta2-ta1}), backward: ({ta3-ta2})")
    

