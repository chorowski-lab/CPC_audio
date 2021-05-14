
import torch
import math
import multiprocessing
import time

def seDistancesToCentroids(vecs, centroids):
    # TODO check
    #print(torch.square(centroids).sum(1).view(1,-1).shape, torch.square(vecs).sum(1).view(-1,1).shape, torch.matmul(vecs, centroids.T).shape)
    return torch.square(centroids).sum(1).view(1,-1) + torch.square(vecs).sum(1).view(-1,1) - 2*torch.matmul(vecs, centroids.T)

def seMinDistancesToCentroidsOneArg(ar):
    vecs, centroids = ar
    return torch.argmin(seDistancesToCentroids(vecs, centroids), 1)

def batchClosest(batch, centroids, lengths, pool, norm=False):
    B = batch.shape[0]
    #print("!!!!!!!!!!!!!!!!!!", batch.dtype, centroids.dtype)
    if norm:
        centroidLengths = torch.sqrt((centroids*centroids).sum(1))
        #print(centroids.shape, centroidLengths.shape, centroidLengths.view(-1,1).shape)
        centroids = centroids / centroidLengths.view(-1,1)
        #print(torch.sqrt((centroids*centroids).sum(1)))
        for i in range(B):
            batchLengths = torch.sqrt((batch[i][:lengths[i]] * batch[i][:lengths[i]]).sum(1))
            #print(batchLengths.shape, batchLengths.view(-1,1).shape)
            batch[i][:lengths[i]] = batch[i][:lengths[i]] / batchLengths.view(-1,1)
            #if i == 0:
            #    print(torch.sqrt((batch[i][:lengths[i]]*batch[i][:lengths[i]]).sum(1)))
    args = [(batch[i][:lengths[i]], centroids) for i in range(B)]
    results = pool.map(seMinDistancesToCentroidsOneArg, args)
    res = torch.full(batch.shape[:2], -1)
    for i in range(B):
        #print("::::", res.shape, results[i], results[i].shape, res[i][:3].shape)
        res[i][:lengths[i]] = results[i]
        
    return res

def prunedHMMpieceSegmentation(piece, centers, shortening, allowedDrift):

    # piece should be len x dim tensor, centers should be num_centers x dim tensor

    emissionCosts = seDistancesToCentroids(piece, centers)
    #print(emissionCosts)

    N = piece.shape[0]
    k = centers.shape[0]

    dyn = torch.full((N, 2*allowedDrift + 1, k), float("Inf"), dtype=float)
    pathK = torch.full((N, 2*allowedDrift + 1, k), -1)
    pathW = torch.full((N, 2*allowedDrift + 1, k), -1)

    #print(dyn.shape, pathK.shape, pathW.shape)

    avgSegmDelta = 1. / shortening

    # drifts will be OK, eg. if shortening = 3., drift will jump exactly after each 3 segments
    # 0 is needed at the beginning in currentAvgSegmNum for that, as avgSegmDelta is added before check and loop starts from 2nd row/column
    currentAvgSegmNum = 0.
    currentNoDriftSegmNum = 0
    middle = allowedDrift
    width = 2*allowedDrift + 1

    dyn[0,middle] = emissionCosts[0]

    # print(f'W: {width}, N: {N}, k: {k}, avgSegmDelta: {avgSegmDelta}')

    for i in range(1, N):

        

        currentAvgSegmNum += avgSegmDelta
        oldNoDriftSegmNum = currentNoDriftSegmNum
        currentNoDriftSegmNum = int(math.floor(currentAvgSegmNum))
        driftDiff = currentNoDriftSegmNum - oldNoDriftSegmNum

        # print("!!!!", driftDiff)
        
        # no-new-segment update
        boringTensor = torch.arange(0, k)
        for d in range(0, width-driftDiff):
            dyn[i, d] = torch.min(dyn[i, d], dyn[i-1, d+driftDiff])
            #print(pathK[i, d].shape, boringTensor.shape)
            pathK[i, d] = boringTensor
            pathW[i, d] = d + driftDiff
        
        #print(dyn[i])

        # TODO think if allow considering aaaaa as possibly more than one segment
        #      rather not, as would then skip any merging in this way, just introducing pseudo-borders
        # segment-change update
        for d in range(max(0, 1-driftDiff), min(width-driftDiff+1, width)):
            prevStep = dyn[i-1, d-1+driftDiff]
            
            bestPrevInfo = torch.topk(prevStep, 2, largest=False)
            bestPrevScores = bestPrevInfo.values
            bestPrevStates = bestPrevInfo.indices
            betterUpdates = dyn[i, d] > bestPrevScores[0]  # prevStep is incorrect, value for correct prevK should only be taken
            # "###" were for not allowing "up to k matches", but "exactly k"
            ###prevIn = dyn[i, d, bestPrevStates[0]].item()  # without that, torch just creates view and doesn;t store value
            ###prevInK = pathK[i, d, bestPrevStates[0]].item()
            ###prevInW = pathW[i, d, bestPrevStates[0]].item()
            # if i > 5:
            #     print("---")
            #     print(":1", bestPrevInfo, prevIn)
            #     print("::", dyn[i,d], prevStep)
            dyn[i, d] = torch.min(dyn[i, d], bestPrevScores[0])
            # if i > 5:
            #     print(":2", dyn[i, d], prevIn)
            #     print("!", betterUpdates)
            pathK[i, d, betterUpdates] = bestPrevStates[0]
            pathW[i, d, betterUpdates] = d-1+driftDiff
            # if i > 5:
            #     print(":3", prevIn, bestPrevScores[1]) 
            #     print('::', dyn[i, d])
            #     print('::', pathK[i, d])
            ###dyn[i, d, bestPrevStates[0]] = min(prevIn, bestPrevScores[1])  # fixing state that was best with 1 segm less - not allowing artificial border
            
            ###if bestPrevScores[1] < prevIn:
            ###    pathK[i, d, bestPrevStates[0]] = bestPrevStates[1]
            ###    pathW[i, d, bestPrevStates[0]] = d-1+driftDiff
            ###else:
            ###    pathK[i, d, bestPrevStates[0]] = prevInK
            ###    pathW[i, d, bestPrevStates[0]] = prevInW
            # if i > 5:
            #     print(":4", dyn[i, d])
            #     print('::', pathK[i, d])

        # print(dyn[i])

        # + emission at the end   TODO check if ok adding to inf
        dyn[i] += emissionCosts[i]
        # print(i, emissionCosts[i])
        # print(dyn[i])
        # print(pathK[i])
        # print(pathW[i])

    costs = torch.full((width,), float("Inf"), dtype=float)
    states = torch.full((width,N), -1)
    currentRealSegmNumNoDrift = currentNoDriftSegmNum + 1
    segmNums = torch.arange(currentRealSegmNumNoDrift - allowedDrift, currentRealSegmNumNoDrift + allowedDrift + 1, dtype=int)
    assert segmNums[middle] == currentRealSegmNumNoDrift
    segmDrifts = torch.arange(width, dtype=int) - middle
    assert segmDrifts[middle] == 0
    for i in range(width):
        endState = torch.argmin(dyn[N-1, i])
        whereK = endState
        whereW = i
        whereN = N-1
        endCost = dyn[whereN, whereW, whereK]
        costs[i] = endCost
        while whereN >= 0:
            states[i, whereN] = whereK
            pK = whereK
            pW = whereW
            whereK = pathK[whereN, pW, pK]
            whereW = pathW[whereN, pW, pK]
            whereN -= 1

    #print("OUT")
    return segmNums, segmDrifts, costs, states
        
def prunedHMMpieceSegmentationOneArg(arg):
    piece, centers, shortening, allowedDrift = arg
    #print("ENTER")
    return prunedHMMpieceSegmentation(piece, centers, shortening, allowedDrift)

def prunedHMMsegmentation(batch, centers, shortening, allowedDrift, allowedWholeDrift, lengths=None, segmBreakBetweenLines=True, pool=None, shortenBatch=None):
    
    pathsLi = []
    k = centers.shape[0]
    B0 = batch.shape[0]
    if lengths is None:
        N = batch.shape[1]
        lengths = [N for i in range(B0)]
    
    
    # if segmBreakBetweenLines:
    #     driftRemainderPerBatch = 1 - ((float(N) / shortening) - math.floor(float(N) / shortening))  
    #     # ^ this is actually negative, but remembered as pos; 1, because we do a segm break between the lines (if we assume that)
    # else:  
    # this actually assumes segm break XD

    batchNotTooLong = []
    newLengths = []
    returnDict = {}
    for i in range(B0):
        if not shortenBatch:
            returnDict[len(batchNotTooLong)] = (i,0,lengths[i])
            batchNotTooLong.append(batch[i][:lengths[i]])
            newLengths.append(lengths[i])
        else:
            #print("A")
            nowLen = 0
            while nowLen < lengths[i]:
                #print(i)
                returnDict[len(batchNotTooLong)] = (i,nowLen,min(lengths[i],nowLen+shortenBatch))
                batchNotTooLong.append(batch[i][nowLen:min(lengths[i],nowLen+shortenBatch)])
                newLengths.append(min(lengths[i],nowLen+shortenBatch) - nowLen)
                nowLen += shortenBatch

    print(f'Adjusted lines lengths: {newLengths}')
    #print(returnDict)
    B = len(batchNotTooLong)
    dynLi = torch.zeros((B, 2*allowedDrift + 1))
    driftsLi = torch.zeros((B, 2*allowedDrift + 1), dtype=int)
    segmsLi = []
    
    #print(batchNotTooLong)

    startTime = time.time()

    if pool is None:
        for i in range(B):

            line = batchNotTooLong[i]

            segmNumsI, segmDriftsI, costsI, statesI = prunedHMMpieceSegmentation(line, centers, shortening, allowedDrift)  # [:lengths[i]]  now already cut

            #print(costsI.shape, segmDriftsI.shape, ":", dynLi.shape, driftsLi.shape)
            driftsLi[i] = segmDriftsI
            pathsLi.append(statesI)
            dynLi[i] = costsI
            segmsLi.append(segmNumsI)
    else:
        args = [(batchNotTooLong[i], centers, shortening, allowedDrift) for i in range(B)]   # now already cut [:lengths[i]]
        results = pool.map(prunedHMMpieceSegmentationOneArg, args)
        
        for i in range(B):
            #print(i)
            segmNumsI, segmDriftsI, costsI, statesI = results[i]

            #print(costsI.shape, segmDriftsI.shape, ":", dynLi.shape, driftsLi.shape)
            driftsLi[i] = segmDriftsI
            pathsLi.append(statesI)
            dynLi[i] = costsI
            segmsLi.append(segmNumsI)

        # not using segmNums now as segmDrifts with driftRemainderPerBatchshould be more convenient; but keep for part od end result

    # print("!!!!\n", pathsLi)

    afterConcurrentTime = time.time()
    print(f'time for multiproc part: {afterConcurrentTime - startTime}')

    dyn = torch.full((B, 2*allowedWholeDrift + 1), float("Inf"), dtype=float)
    pathDyn = torch.full((B, 2*allowedWholeDrift + 1), -1)
    bestLinePath = torch.full((B, 2*allowedWholeDrift + 1), -1)

    middle = allowedWholeDrift

    dyn[0, (allowedWholeDrift-allowedDrift):(allowedWholeDrift+allowedDrift+1)] = dynLi[0]
    bestLinePath[0, (allowedWholeDrift-allowedDrift):(allowedWholeDrift+allowedDrift+1)] = torch.arange(2*allowedDrift+1, dtype=int)
    width = 2*allowedWholeDrift + 1

    #print(dynLi[0])

    # driftRemainderNow = 0.  #driftRemainderPerBatchAbs  # here we start from that as there is no additional segment at start
    # if segmsLi[0][allowedDrift].item() - (float(N) / shortening) > 0:  # in this case if we want consistent bahaviour (<=segments than division float), we need to make out floor ceiling in a way
    #     driftRemainderNow += 1.  
    allDriftNow = segmsLi[0][allowedDrift].item() - (float(newLengths[0]) / shortening)
    # that was assumed to be tensor and shared, no comm
    # print("!", driftRemainderPerBatchAbs*driftSign)

    # print("====", dyn[0])
    # print("____", bestLinePath[0])
    # print("______________________________")
    for i in range(1, B):
        
        # each length the same, so just taking from 0
        driftRemainderThisBatch = segmsLi[i][allowedDrift].item() - (float(newLengths[i]) / shortening)   # we are making >= segms than we should, not <=
        allDriftNow += driftRemainderThisBatch
        driftDiff = 0
        while allDriftNow < -1:
            driftDiff -= 1
            allDriftNow += 1.
        while allDriftNow > 0:
            driftDiff += 1
            allDriftNow -= 1.
        #driftSign = int(torch.sign(torch.tensor([driftRemainderPerBatch])).item())
        #driftRemainderPerBatchAbs = abs(driftRemainderPerBatch)
        
        #driftRemainderNow += driftRemainderPerBatchAbs
        # print("******", i, driftRemainderNow, ":", driftRemainderPerBatchAbs*driftSign)
        # driftDiff = int(math.floor(driftRemainderNow))
        # driftRemainderNow -= math.floor(driftRemainderNow)
        # if segmBreakBetweenLines:
        #     driftDiff = -driftDiff  # in this case drift is negative and was counted as positive for floor to work ok
        drifts = driftsLi[i] + driftDiff  #*driftSign
        # print("!", drifts, drifts.dtype, driftDiff)
        # print("----", pathDyn[i])
        # print("____", bestLinePath[i])
        for w in range(width):
            minDrift = drifts[0].item()
            maxDrift = drifts[-1].item()
            #print(";", minDrift, maxDrift, w, width)
            startDriftIdx = max(-(w + minDrift), 0)
            firstOutDriftIdx = max(drifts.shape[0] - max((w + maxDrift) - width + 1, 0), 0)   # first out; TODO check
            #print(startDriftIdx, firstOutDriftIdx)
            startDrift = drifts[startDriftIdx].item()
            firstOutDrift = drifts[firstOutDriftIdx - 1].item() + 1  # +1 to be out drift, not last ok
            #print(drifts, startDrift, firstOutDrift, "[", w+startDrift, w+firstOutDrift)
            newScores = dyn[i-1, w] + dynLi[i, startDriftIdx:firstOutDriftIdx]
            oldScores = dyn[i, (w+startDrift):(w+firstOutDrift)]
            #print("!::", oldScores.shape, newScores.shape, w, drifts, startDrift, firstOutDrift)
            better = newScores < oldScores
            dyn[i, (w+startDrift):(w+firstOutDrift)] = torch.min(dyn[i, (w+startDrift):(w+firstOutDrift)], newScores)
            pathDyn[i, (w+startDrift):(w+firstOutDrift)][better] = w  # TODO check if works as wanted
            bestLinePath[i, (w+startDrift):(w+firstOutDrift)][better] = torch.arange(startDriftIdx, firstOutDriftIdx)[better]
        # print("----", pathDyn[i])
        # print("____", bestLinePath[i])
        # print("====", dyn[i])

    # whole path; could do for all drifts in range, but meh
    chosenInLines = torch.full((B,), -1)
    
    endCost = dyn[B-1, middle]
    whereW = middle
    # print(middle)
    whereN = B-1
    # print("####", dyn[B-1], whereW, whereN)
    # print(dyn)
    # print(pathDyn)
    # print(bestLinePath)
    
    while whereN >= 0:
        
        chosenInLines[whereN] = bestLinePath[whereN, whereW].item()
        # print("i", whereW, bestLinePath[whereN, whereW].item(), chosenInLines[whereN], chosenInLines)
        whereW = pathDyn[whereN, whereW].item()
        whereN -= 1

    endTime = time.time()
    print(f'time for merging part: {endTime - afterConcurrentTime}')

    # print("))))))))))))))))",  chosenInLines)
    # print("************", pathsLi)
    # print("************", segmsLi)
    allPaths = torch.full((B0, max(lengths)), -1)
    numsSegmsInLines = torch.full((B0,), 0)
    for i in range(B):
        line, begin, end = returnDict[i]
        # allPaths[i][:lengths[i]] = pathsLi[i][chosenInLines[i].item()]
        # numsSegmsInLines[i] = segmsLi[i][chosenInLines[i].item()]
        allPaths[line][begin:end] = pathsLi[i][chosenInLines[i].item()]
        numsSegmsInLines[line] += segmsLi[i][chosenInLines[i].item()]
    #allPaths = allPaths.view(-1).contiguous()  rather not needed

    

    return numsSegmsInLines, allPaths







if __name__ == '__main__':
    print("B")
    line = torch.tensor([[1],[1],[2]]) #,[2],[2],[2],[3],[3],[3]])
    centers = torch.tensor([[1],[2]])  #,[3]])
    line = torch.tensor([[1],[1],[2],[2],[2],[2],[3],[3],[3]])
    centers = torch.tensor([[1],[2],[3]])
    print(line.shape, centers.shape)

    print(prunedHMMpieceSegmentation(line, centers, 3., 1))

    line2 = torch.tensor([[1],[1],[2]], dtype=float)
    centers2 = torch.tensor([[1],[2]], dtype=float)
    lines2 = torch.zeros(3, *(line2.shape), dtype=float)
    for i in range(3):
        lines2[i] = line2
    print(lines2.shape, centers2.shape)
    print(prunedHMMsegmentation(lines2, centers2, 1.51, 1, 5, pool=multiprocessing.Pool(10)))
    # 2.25 is the border between 3 and 4 segments, 1.8 is between 4 and 5, 1.5 between 5 and 6
    #print(prunedHMMsegmentation(lines2, centers2, 2., 1))

    line3 = torch.tensor([[1],[1],[1],[1],[2]], dtype=float)
    centers3 = torch.tensor([[1],[2]], dtype=float)
    lines3 = torch.zeros(3, *(line3.shape), dtype=float)
    lines3[0] = torch.tensor([[1],[1],[2],[-1],[-1]], dtype=float)
    lines3[1] = torch.tensor([[1],[1],[1],[1],[2]], dtype=float)
    lines3[2] = torch.tensor([[1],[1],[1],[2],[-1]], dtype=float)
    print(lines3.shape, centers3.shape)
    print(prunedHMMsegmentation(lines3, centers3, 2.01, 1, 5, lengths=[3,5,4], pool=multiprocessing.Pool(10)))
    # here borders are (bigger sum of lengths):  3 ---3.0--- 4 ---2.4--- 5 ---2.0--- 6 ---1.71---  7

    centers8 = torch.tensor([[1],[2]], dtype=float)
    lines8 = torch.zeros(2, 9, 1, dtype=float)
    lines8[0] = torch.tensor([[1],[1],[1],[2],[-1],[-1],[-1],[-1],[-1]], dtype=float)
    lines8[1] = torch.tensor([[1],[1],[1],[1],[2],[1],[1],[2],[-1]], dtype=float)
    print(lines8.shape, centers8.shape)
    print(prunedHMMsegmentation(lines8, centers8, 2.01, 1, 5, lengths=[4,8], pool=multiprocessing.Pool(10), shortenBatch=5))
    # here borders are (bigger sum of lengths):  3 ---3.0--- 4 ---2.4--- 5 ---2.0--- 6 ---1.71---  7




    line4 = torch.tensor([[1],[1],[1],[1],[2]], dtype=float)
    centers4 = torch.tensor([[1],[2]], dtype=float)
    lines4 = torch.zeros(3, *(line4.shape), dtype=float)
    lines4[0] = torch.tensor([[1],[1],[2],[-1],[-1]])
    lines4[1] = torch.tensor([[1],[1],[1],[1],[2]])
    lines4[2] = torch.tensor([[1],[1],[1],[2],[-1]])
    print(lines4.shape, centers4.shape)
    print(batchClosest(lines4, centers4, [3,5,4], multiprocessing.Pool(10)))
    