
import torch
import numpy as np
from cpc.segm.segment_dict import *
from heapq import *
from torch.autograd import Function, Variable
import time

def variance(linearSum, squaresSum, size):
    return np.sum((squaresSum / size) - np.square(linearSum / size))  # sum of "variance mse vector"

def se(linearSum, squaresSum, size):  # square error
    return np.sum(squaresSum - np.square(linearSum) / size)  # sum of "se vector"

def varianceDiff(stats1, stats2):
    linearSum1, squaresSum1, size1 = stats1
    linearSum2, squaresSum2, size2 = stats2
    return variance(linearSum1 + linearSum2, squaresSum1 + squaresSum2, size1 + size2) - variance(linearSum1, squaresSum1, size1) - variance(linearSum2, squaresSum2, size2)

def seDiff(stats1, stats2):
    linearSum1, squaresSum1, size1 = stats1
    linearSum2, squaresSum2, size2 = stats2
    return se(linearSum1 + linearSum2, squaresSum1 + squaresSum2, size1 + size2) - se(linearSum1, squaresSum1, size1) - se(linearSum2, squaresSum2, size2)

def cosDist(stats1, stats2):  # cosine distance
    linearSum1, size1 = stats1
    linearSum2, size2 = stats2
    unscaledSim = np.dot(linearSum1, linearSum2) / (np.sqrt(np.dot(linearSum1, linearSum1)) * np.sqrt(np.dot(linearSum2, linearSum2)))
    unscaledAsDist = -unscaledSim + 1.  # change from similarity to distance; we mainly care about order for priority queue, for that any mapping reversing order is ok (low similarity = high distance)
    # ^ here we have a change from [-1, 1] to [0, 2]; standard "cosine distance"
    return unscaledAsDist * (size1 + size2)  
    # scaling so that big nonsense averaged almost-random segments don't appear as similar (randomnoise1 ~= randomnoise2)
    # this is where changing form similarity to distance mapping can make a difference, but linear one seems ok
    # this scaling is similar to the sum of distances of all elements to the average of the another segment and vice versa (can use sums instead of averages for cosine sim; 
    # but that's perhaps not exactly this sum as cosine_similarity ( (sum_i a_i) , x ) is not the same as (sum_i cosine_similarity ( a_i , x )) )
    # but the other one would be more expensive to compute

def seDistancesToCentroids(vec, centroids):
    return np.square(centroids).sum(1) + np.square(vec).sum() - 2*np.matmul(centroids, vec)  # TODO? can later do this in GPU for all at once or so (in segmentsDict e.g.)

# to be later used after completed segmentation on final segments
def closestSeCentroid(vec, centroids):
    return np.argmin(seDistancesToCentroids(vec, centroids))

def centroidSeDiff(stats1, stats2):  # this doesn't perhaps actually need centroids, only stuff computing begin len-1 stats needs that perhaps
    dists1 = stats1[0]
    dists2 = stats2[0]
    return np.min(dists1 + dists2) - np.min(dists1) - np.min(dists2)

# TODO option for rounding-to-prototype loss for kmeans option

def linRoundingLoss(mean, originals):
    return torch.abs(originals - mean).sum()

def varRoundingLoss(mean, originals):
    return torch.mean(torch.square(originals - mean), dim=0).sum()

def seRoundingLoss(mean, originals):
    return torch.square(originals - mean).sum()

def cosRoundingLoss(mean, originals):
    #print(mean.shape, mean.T.shape, originals.shape, torch.matmul(originals, mean).shape, "!", torch.sqrt(torch.dot(mean, mean)).shape, torch.sqrt(torch.matmul(originals, originals.T)).diagonal().shape)
    unscaledSim = torch.matmul(originals, mean) / (torch.sqrt(torch.dot(mean, mean)) * torch.sqrt(torch.matmul(originals, originals.T)).diagonal())
    #print(":::", unscaledSim.shape, (-unscaledSim + 1).sum())
    # if unscaledSim.min() < 0:
    #     print(unscaledSim)
    unscaledAsDist = -unscaledSim + 1.
    return unscaledAsDist.sum()

def segmDictToIntTensor(segmDict):
    size = len(segmDict) * 5
    tens = torch.zeros(size, dtype=int)
    where = 0
    for line, idxInLine in segmDict.keys():
        line2, begin, end = segmDict[(line, idxInLine)]
        tens[where] = line
        tens[where+1] = idxInLine
        tens[where+2] = line2
        tens[where+3] = begin
        tens[where+4] = end
        where += 5
    return tens

def intTensorToSegmDict(tens):
    tens = tens.cpu()
    dct = {}
    i = 0
    while i < tens.shape[0]:
        line = tens[i]
        idxInLine = tens[i+1]
        line2 = tens[i+2]
        begin = tens[i+3]
        end = tens[i+4]
        dct[(line, idxInLine)] = (line2, begin, end)
        i += 5
    return dct

def mergeStats(segmDictTens, label, numPhones):
    label = label.cpu()
    segmDict = intTensorToSegmDict(segmDictTens)
    merges = torch.zeros(numPhones, numPhones, dtype=torch.float32).cpu()
    counts = torch.zeros(numPhones, dtype=torch.float32).cpu()
    for line, idxInLine in segmDict.keys():
        line2, begin, end = segmDict[(line, idxInLine)]
        labelsThere = list(map(lambda x: x.item(), label[line2, begin:(end+1)]))
        for x in labelsThere:
            counts[x] += 1
            for y in labelsThere:
                merges[x,y] += 1
            merges[x,x] -= 1
    return merges, counts
                

# [!] lines has to be a numpy array, np.sum() crashes if done on tensor
def hierarchicalSegmentation(lines, padMask=None, k=None, minSegmsPerLine=None, mergePriority="se", centroids=None):  # k is sum of number of segments for all lines
    
    t0 = time.time()

    if mergePriority == "se":  # var not divided by size, square error
        segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine,
                                    segmStats = [lambda x: x, lambda x: np.square(x), lambda x: 1],  # for lengths 1
                                    segmStatsMerges = [lambda x, y: x + y, lambda x, y: x + y, lambda x, y: x + y])
        costFun = seDiff  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: seDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
        len1costFun = lambda x : 0
    elif mergePriority == "var":  # var is mse
        segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine,
                                    segmStats = [lambda x: x, lambda x: np.square(x), lambda x: 1],  # for lengths 1
                                    segmStatsMerges = [lambda x, y: x + y, lambda x, y: x + y, lambda x, y: x + y])
        costFun = varianceDiff  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: varianceDiff(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
        len1costFun = lambda x : 0
    elif mergePriority == "cos":
        segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine,
                                    segmStats = [lambda x: x, lambda x: 1],  # for lengths 1
                                    segmStatsMerges = [lambda x, y: x + y, lambda x, y: x + y])
        costFun = cosDist  #lambda linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2: cos(linearSum1, squaresSum1, size1, linearSum2, squaresSum2, size2)
        len1costFun = lambda x : 0
    elif mergePriority == "centroid-se":
        assert centroids is not None
        # TODO ! here and also in all other options use SegmentDict with chosen stats to compute (see TODOs in SegmentDict) 
        # TODO here below instead of dct.getSegmentSums have a dct.getSegmentStats returned AND THEN feed this to costFun (add different args too - which will have different arguments for se&var/cos/centroids)
        segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine,
                                    segmStats = [(lambda centroids: (lambda vec: seDistancesToCentroids(vec, centroids)))(centroids)],  # for lengths 1
                                    segmStatsMerges = [lambda x, y: x + y])
        costFun = centroidSeDiff   #(lambda ctr: centroidSeDiff)(centroids)
        len1costFun = lambda x : np.min(x)
    elif mergePriority == "centroid-cos":
        assert centroids is not None
        assert False   # TODO
    else:
        assert False

    ###segmentsDict = SegmentDict(lines, padMask=padMask, minSegmsPerLine=minSegmsPerLine)
    #--t1 = time.time()
    # maybe will need to change this to arrays or so instead of dicts for efficiency
    
    # q for ranges to merge
    q = []
    
    len1cost = 0.
    
    # every pair added only one time; after merges will need to add both to right and to left
    for segm in segmentsDict.getSegments():
        stats1 = segmentsDict.getSegmentStats(segm)
        len1cost += len1costFun(stats1)
        segmRight = segmentsDict.getSegmentRight(segm)
        if segmRight is not None:
            #print(segm, segmRight)
            #print(stats1)
            #print("_____________")
            stats2 = segmentsDict.getSegmentStats(segmRight)
            ###line1, left1, right1 = segm
            ###line2, left2, right2 = segmRight
            #oldVar1 = costFun(linSum1, sqSum1, right1 - left1 + 1)
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSum1 + linSum2, sqSum1 + sqSum2, right2 - left1 + 1)
            ###size1 = right1 - left1 + 1
            ###size2 = right2 - left2 + 1
            costDiff = costFun(stats1, stats2)  #linSum1, sqSum1, size1, linSum2, sqSum2, size2)
            heappush(q, (costDiff, segm, segmRight))
       
    costSum = len1cost

    varChanges = []
    merges = []

    #--t2 = time.time()
    
    while len(q) and (k is None or segmentsDict.numSegments() > k):  # will stop merging before k reached if minSegmsPerLine reached
    
        varChange, left, right = heappop(q)
        merged = segmentsDict.mergeSegments(left, right)  # checks if merge is valid
        
        if merged is None:  # old merge possibility, now impossible (or minSegmsPerLine reached for this line)
            continue
        
        varChanges.append(varChange)
        costSum += varChange
        merges.append((left, right))
        
        toLeft = segmentsDict.getSegmentLeft(merged)
        toRight = segmentsDict.getSegmentRight(merged)
        statsMerged = segmentsDict.getSegmentStats(merged)
        lineMerged, leftMerged, rightMerged = merged
        sizeMerged = rightMerged - leftMerged + 1
        #varMerged = costFun(linSumMerged, sqSumMerged, rightMerged - leftMerged + 1)
        
        if toLeft is not None:
            stats2 = segmentsDict.getSegmentStats(toLeft)
            line2, left2, right2 = toLeft
            size2 = right2 - left2 + 1
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSumMerged + linSum2, sqSumMerged + sqSum2, rightMerged - left2 + 1)
            costDiff = costFun(statsMerged, stats2)  #linSumMerged, sqSumMerged, sizeMerged, linSum2, sqSum2, size2)
            heappush(q, (costDiff, toLeft, merged))
            
        if toRight is not None:
            stats2 = segmentsDict.getSegmentStats(toRight)
            line2, left2, right2 = toRight
            size2 = right2 - left2 + 1
            #oldVar2 = costFun(linSum2, sqSum2, right2 - left2 + 1)
            #mergedVariance = costFun(linSumMerged + linSum2, sqSumMerged + sqSum2, right2 - leftMerged + 1)
            costDiff = costFun(statsMerged, stats2)  #linSumMerged, sqSumMerged, sizeMerged, linSum2, sqSum2, size2)
            heappush(q, (costDiff, merged, toRight))
            
    #--t3 = time.time()
    #--print(f"inside merging times: {t1-t0}, {t2-t1}, {t3-t2}")

    return (len1cost, costSum, varChanges), merges, segmentsDict

class HierarchicalSegmentationLayer(Function):

    @staticmethod
    def flatten(x):
        s = x.shape()
        if len(s) < 3:
            return x
        if len(s) == 3:
            return x.view(-1, s[2])
        assert False

    # perhaps that ^ is not needed, and restore_shapes also
    
    @staticmethod
    def getKforGivenShorteningAndShape(shape, shortening):
        numReprs = float(np.prod(shape[:-1]))
        return max(int(round(numReprs / float(shortening))), 1)

    @staticmethod
    def forward(
        ctx, 
        inputGPU, 
        padMask=None, 
        k=None, 
        allowKsumRange=None, 
        minSegmsPerLine=None, 
        mergePriority="se", 
        shorteningPolicy="orig_len", 
        roundingLossType=None, 
        centroids=None): 
    # k for strict num of segments (SUM FOR ALL LINES), allowKsumRange for range OF SUM OF SEGMENTS IN ALL LINES and choosing 'best' split point
    # min and max number of merges adjusted to what is possible - e.g. because of minSegmsPerLine

        #--t0 = time.time()

        assert k is None or allowKsumRange is None  # mutually exclusive options
        assert shorteningPolicy in ("shorten", "orig_len")  # orig_len+guess_orig is only at the higher level
        assert roundingLossType in ("se", "var", "lin", "cos", None)
        if roundingLossType == "se":  
            roundingLossFun = seRoundingLoss  
        elif roundingLossType == "var":  
            roundingLossFun = varRoundingLoss 
        elif roundingLossType == "lin":  
            roundingLossFun = linRoundingLoss  
        elif roundingLossType == "cos":
            roundingLossFun = cosRoundingLoss  
        else:
            assert roundingLossType is None
            roundingLossFun = None

        # TODO if input only 2-dim, add another dimension possibly (W x H -> 1 x W x H, consistent with B x W x H - later assuming that in some places)

        inputDevice = inputGPU.device
        padMaskInputDevice = padMask.device if padMask is not None else False

        # tensor to CPU  (don't really need copy, will just need to put tensors in segmentsDict)
        input = inputGPU.detach().to('cpu').numpy()  
        # https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301 ,
        # https://discuss.pytorch.org/t/what-is-the-cpu-in-pytorch/15007/3

        costInfo, merges, segmentsDict = hierarchicalSegmentation(input, padMask=padMask, k=k, minSegmsPerLine=minSegmsPerLine, mergePriority=mergePriority, centroids=centroids)  # won't modify input
        _1, _2, varChanges = costInfo  
        #print("MERGES0: ", merges)
        if allowKsumRange:  # full merge done above, k=None, so each line now has minSegmsPerLine, but can also just get it from SegmDict - cleaner
            begin, end = allowKsumRange
            assert begin <= end
            # [!] min and max number of merges adjusted to what is possible - e.g. because of minSegmsPerLine
            beginIdx = max(0, min(len(varChanges) - 1, (segmentsDict.numSegments() + (len(varChanges) - 1) - end)))  # max allowed num of segments, smallest num of merges; input.shape[0] is num of segments if all merges done
            endIdx = max(0, min(len(varChanges) - 1, (segmentsDict.numSegments() + (len(varChanges) - 1) - begin)))  # min allowed num of segments, biggest num of merges; input.shape[0] is num of segments if all merges done
            #print("::::::::::", beginIdx, endIdx)
            prefSums = []
            s = 0.
            for chng in varChanges:
                s += chng
                prefSums.append(s)
            best = -1
            where = -1
            #print("PREFSUMS: ", prefSums)
            for i in range(beginIdx, min(endIdx+1, len(varChanges))):
                sufSum = s - prefSums[i]  # sum after this index
                prefSum = prefSums[i] if prefSums[i] > 0. else .0000001  # don't div by 0
                # v the bigger the better split point; suffix div by prefix averages of variance change
                here = (sufSum / (len(varChanges)-i))  /  (prefSum / (i+1.))  
                #print("!", i, ":", prefSum ,sufSum, here)
                
                if here > best:
                    best = here
                    where = i
            if where == -1:
                print("WARNING: problems choosing best num segments")
                where = int((beginIdx + endIdx) // 2)
            varChanges = varChanges[:where+1]  # this one is not really needed
            merges = merges[:where+1]
            
        finalSegments, segmentNumsInLines = SegmentDict.getFinalSegments(merges, input.shape[:2], padMask=padMask)
        #print("MERGES: ", merges)
        #print("FINAL SEGMENTS: ", finalSegments)

        #--t1 = time.time()

        maxSegments = max(segmentNumsInLines)
        
        if shorteningPolicy == "shorten":
            segmented = np.full((input.shape[0], maxSegments, input.shape[2]), 0.)  #torch.tensor(size=(input.shape[0], maxSegments, input.shape[2])).fill_(0.)
            if padMask is not None:
                paddingMaskOut = np.full((input.shape[0], maxSegments), False)  #torch.BoolTensor(size=(input.shape[0], maxSegments)).fill_(False)
                for i, n in enumerate(segmentNumsInLines):
                    paddingMaskOut[i][n:] = True
                resPadMask = torch.BoolTensor(paddingMaskOut).to(padMaskInputDevice)
        else:
            segmented = np.full(input.shape, 0.)
            if padMask is not None:
                resPadMask = padMask
        if padMask is None:
            resPadMask = torch.zeros(1).to(inputDevice)
        # can perhaps return a tensor with 1 at the beginning of the segments, -1 at the end, 0s elsewhere
        segmentBorders = np.zeros((input.shape[0], input.shape[1]), dtype=np.int8)
        roundingLoss = torch.tensor(0, dtype=torch.float32).requires_grad_(True).to(inputDevice)  # TODO dtype (?)
        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            if shorteningPolicy == "shorten":
                segmented[line][idxInLine] = np.mean(input[line][begin:(end+1)], axis=0)  #torch.mean(input[line][begin:(end+1)])
            else:
                segmented[line][begin:(end+1)] = np.mean(input[line][begin:(end+1)], axis=0)
            if roundingLossFun is not None:
                roundingLoss += roundingLossFun(torch.mean(input[line][begin:(end+1)], dim=0), input[line][begin:(end+1)])
            segmentBorders[line][end] = -1  
            segmentBorders[line][begin] = 1  # [!] can be e.g. [...0, 0, 1, 1, ...] with segment of length 1 
            # - marking begins when length 1 as * scaling doesn't need + (scale-1) there if logging only begins

        resOutput = torch.tensor(segmented, dtype=inputGPU.dtype).to(inputDevice)   #if wasInputOnGPU else torch.tensor(segmented)  #.requires_grad_(True)
        # resPadMask created above, as for some reason torch.BoolTensor(paddingMaskOut).to(padMaskInputDevice) thrown an error if paddingMaskOut was a tensor on a correct device
        segmentBorders = torch.IntTensor(segmentBorders).to(inputDevice)

        #print("********************", dir(ctx))
        #[not really needed] ctx.save_for_backward(padMask, resPadMask)
        # save_for_backward is only for tensors / variables / stuff
        if shorteningPolicy == "shorten":
            ctx.shortened = True
        else:
            ctx.shortened = False
        ctx.finalSegments = finalSegments
        ctx.segmentNumsInLines = segmentNumsInLines
        ctx.inputShape = input.shape
        ctx.mark_non_differentiable(resPadMask)  # can only pass torch variables here and only that makes sense

        #--t2 = time.time()

        #--print(f"hier segm time: merging {t1 - t0}, rest {t2 - t1}")
        #--print(f"segments: {sum(segmentNumsInLines)}, {segmentNumsInLines}")

        #print("FINAL SEGMENTS: ", finalSegments, segmentNumsInLines)

        # with rounding loss None, will just return 0
        return resOutput, resPadMask, segmentBorders, roundingLoss, segmDictToIntTensor(finalSegments)  #, finalSegments, segmentNumsInLines can only return torch variables... TODO maybe check how to fetch this info, but not sure if needed

    @staticmethod
    def backward(ctx, dxThrough, outPadMask=None, segmentBorders=None, roundingLoss=None, finalSegmentsAsTens=None):  #, finalSegments=None, segmentNumsInLines=None):

        dxThroughDevice = dxThrough.device

        #[not really needed] paddingMask, paddingMaskOut = ctx.saved_tensors
        dx = torch.empty(size=ctx.inputShape, dtype=dxThrough.dtype).fill_(0.).to('cpu')

        wasShortened = ctx.shortened
        dxThrough = dxThrough.detach().cpu()
        #print(f"was shortened: {wasShortened}")

        for line, idxInLine in ctx.finalSegments.keys():
            line, begin, end = ctx.finalSegments[(line, idxInLine)]
            #print("!", line, idxInLine, begin, end, dxThrough[line][idxInLine])
            if wasShortened:
                dx[line][begin:(end+1)] = dxThrough[line][idxInLine] / (end - begin + 1)
            else:
                dx[line][begin:(end+1)] = (dxThrough[line][begin:(end+1)].sum(dim=0)) / (end - begin + 1)

        dx = dx.to(dxThroughDevice)

        return dx, None, None, None, None, None, None, None, None


class HierarchicalSegmentationRestoreLengthLayer(Function):

    # @staticmethod
    # def flatten(x):
    #     s = x.shape()
    #     if len(s) < 3:
    #         return x
    #     if len(s) == 3:
    #         return x.view(-1, s[2])
    #     assert False

    # perhaps that ^ is not needed, and restore_shapes also

    # TODO think about padMask if needed, for now assuming there is NO pad mask
    #      yes, will then compute AR on things longer than needed (but can still prune to max len in line AND hier layer DOES that)
    #      and just ignore results out of correct dimensionality

    @staticmethod
    def forward(ctx, shortenedInputGPU, finalSegmentsAsTens): 
    
        finalSegments = intTensorToSegmDict(finalSegmentsAsTens)

        #TODO
        # save finalSegments in ctx
        ctx.finalSegments = finalSegments
        # save input shape in ctx
        ctx.shrinkedShape = shortenedInputGPU.shape
        # similar as main-hier's backward
        device = shortenedInputGPU.device  # this can have smaller dim than original thing
        # NEED to restore original dim - done below v
        originalLen = 0
        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            originalLen = max(originalLen, end+1)

        shortenedInput = shortenedInputGPU.detach().cpu()

        #[not really needed] paddingMask, paddingMaskOut = ctx.saved_tensors
        restored = torch.zeros(size=(shortenedInput.shape[0], originalLen, shortenedInput.shape[2]), dtype=shortenedInput.dtype).to('cpu')

        #wasShortened = ctx.shortened

        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            #if wasShortened:  # TODO? for now only shortening mode
            restored[line][begin:(end+1)] = shortenedInput[line][idxInLine] #/ (end - begin + 1)
            #else:
            #    dx[line][begin:(end+1)] = (dxThrough[line][begin:(end+1)].sum(dim=0)) / (end - begin + 1)

        restored = restored.to(device)

        return restored  #, resPadMask, segmentBorders, roundingLoss, finalSegments  #, finalSegments, segmentNumsInLines can only return torch variables... TODO maybe check how to fetch this info, but not sure if needed

    @staticmethod
    def backward(ctx, dxThrough):  #, outPadMask=None, segmentBorders=None, roundingLoss=None, finalSegments=None):  #, finalSegments=None, segmentNumsInLines=None):

        #TODO

        finalSegments = ctx.finalSegments
        device = dxThrough.device  # this can have smaller dim than original thing
        # NEED to restore original dim - done below v
        shrinkedLen = 0
        for line, idxInLine in finalSegments.keys():
            #line, begin, end = finalSegments[(line, idxInLine)]
            shrinkedLen = max(shrinkedLen, idxInLine+1)

        #[not really needed] paddingMask, paddingMaskOut = ctx.saved_tensors
        dxThrough = dxThrough.detach().cpu()
        reshrinkeddx = torch.zeros(size=ctx.shrinkedShape, dtype=dxThrough.dtype).to('cpu')

        #wasShortened = ctx.shortened

        for line, idxInLine in finalSegments.keys():
            line, begin, end = finalSegments[(line, idxInLine)]
            #if wasShortened:
            reshrinkeddx[line][idxInLine] = dxThrough[line][begin:(end+1)].sum(dim=0)  #/ (end - begin + 1)
            #else:
            #    dx[line][begin:(end+1)] = (dxThrough[line][begin:(end+1)].sum(dim=0)) / (end - begin + 1)

        reshrinkeddx = reshrinkeddx.to(device)

        return reshrinkeddx, None


if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    # run from .. with python -m segmentation.hierarchical_variance_segmentation

    tensor = torch.tensor([[[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]], [[1,2],[1,2],[3,4],[3,4],[3,4],[8,9],[8,9]]], dtype=torch.float64).requires_grad_(True)
    print(tensor[0][1])
    print(hierarchicalSegmentation(tensor.detach().numpy(), padMask=None, k=4, minSegmsPerLine=None, mergePriority="se"))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'
    print(hierarchicalSegmentation(tensor.detach().numpy(), padMask=None, k=2, minSegmsPerLine=None, mergePriority="var"))  # pre-last merge in each line (merging (0,1) and (2,4)) should be 1.92 if summing 'variance vectors'

    print("-------------------------- torch ---------------------------")
    # (tensor, padMask, k, kSumRange)
    resOutput, resPadMask, borders, roundingLoss, dictTens = HierarchicalSegmentationLayer.apply(
        tensor, 
        torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 
        None, 
        (2,5), 
        None, 
        "var", 
        "shorten", 
        None,
        None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch2 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders, roundingLoss, dictTens = HierarchicalSegmentationLayer.apply(tensor, 
    torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, None, "se", "shorten", None, None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch3 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders, roundingLoss, dictTens = HierarchicalSegmentationLayer.apply(tensor, 
    torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, 2, "se", "shorten", None, None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    # [!] here will return 4 segments instead of specified 3, because of specified minSegmsPerLine

    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)

    print("-------------------------- torch4 ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders, roundingLoss, dictTens = HierarchicalSegmentationLayer.apply(tensor, 
    torch.tensor([[True, False, False, False, False, False, False], [False, False, False, False, False, False, True]]), 3, None, 2, "se", "orig_len", None, None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    # [!] here will return 4 segments instead of specified 3, because of specified minSegmsPerLine

    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)








    print("-------------------------- torch with shorten and restore ---------------------------")
    # (tensor, padMask, k, kSumRange)
    tensor.grad.data.zero_()
    resOutput, resPadMask, borders, roundingLoss, dictTens = HierarchicalSegmentationLayer.apply(
        tensor, 
        None, 
        5, 
        None,   # could also have range here
        2, 
        "se", 
        "shorten", 
        None,
        None)  #(2, 5))  # can;t have keyword args for torch Functions...
    print(resOutput)
    print(resPadMask)
    print(borders)
    #print(finalSegments)
    #print(segmentNumsInLines)
    #loss = Variable(resOutput, requires_grad=True)
    resOutput.sum().backward()  # .backward() needs loss to be a number (tensor of size (1,))
    print(tensor.grad)
    tensor.grad.data.zero_()
    print("---")
    resOutRestored = HierarchicalSegmentationRestoreLengthLayer.apply(resOutput, dictTens)
    print(resOutRestored)
    resOutput.sum().backward()
    print(tensor.grad)