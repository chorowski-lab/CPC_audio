
import numpy as np

class SegmentDict:
    
    def __init__(self, lines, padMask=None, minSegmsPerLine=None, segmStats=[], segmStatsMerges=[]):  # lines assumed to be of shape [#lines x line_len * k]
        assert len(segmStats) == len(segmStatsMerges)
        # segm stats take just 1 timestep
        self._segmStats = segmStats
        self._segmStatsMerges = segmStatsMerges
        # (line#, place in line): (begin in line, end in line, sum(x), sum(x^2)) ; sums are possibly vectors
        #self._dct = {(i, j): (j, j, lines[i][j], np.square(lines[i][j])) for i in range(len(lines)) for j in range(len(lines[i])) if padMask is None or not padMask[i][j]}
        self._dct = {(i, j): (j, j, tuple(segmStat(lines[i][j]) for segmStat in self._segmStats)) for i in range(len(lines)) for j in range(len(lines[i])) if padMask is None or not padMask[i][j]}
        self._size = len(self._dct)  # sometimes 1 (now all segments have 1 entry), sometimes later 2 entries per segment - better keep a counter
        # TODO: self._segmStats - array of functions computing stats for points (to enable distances from centroids)
        # TODO: self._segmStatsMerges - array of corresponding to ^ functions telling how to merge stats when merging segments
        # TODO: costFun in hierarchical_segmentation using that
        

        self._line_segms = [0 for i in range(lines.shape[0])]
        for line, _ in self._dct:
            self._line_segms[line] += 1

        self.minSegmsPerLine = minSegmsPerLine
        
    # there is a 'segment' implicit format (tuple) used: (line#, leftIndex(begin), rightIndex(end))
        
    def numSegments(self):
        return self._size
    
    def segmentInDict(self, segment):
        line, leftIdx, rightIdx = segment
        if (line, leftIdx) not in self._dct:
            return False
        leftIdxFromDict, rightIdxFromDict, _ = self._dct[(line, leftIdx)]
        return leftIdx == leftIdxFromDict and rightIdx == rightIdxFromDict  
        # (line, leftIdx) in dct can be for right-range leftIdx with different leftIdxFromDict for merged segment
        
    def removeSegment(self, segment):
        line, leftIdx, rightIdx = segment
        wasThere = False
        if (line, leftIdx) in self._dct:
            del self._dct[(line, leftIdx)]
            wasThere = True
        if (line, rightIdx) in self._dct:
            del self._dct[(line, rightIdx)]
            wasThere = True
        if wasThere:
            self._size -= 1
            self._line_segms[line] -= 1
            
    def mergeSegments(self, segment1, segment2):
        line1, left1, right1 = segment1
        line2, left2, right2 = segment2
        if not self.segmentInDict(segment1) or not self.segmentInDict(segment2) \
           or line1 != line2 or right1 + 1 != left2 \
           or (self.minSegmsPerLine and self._line_segms[line1] <= self.minSegmsPerLine):  
           # not subsequent or too few segments in line
            return None
        stats1 = self.getSegmentStats(segment1)
        stats2 = self.getSegmentStats(segment2)
        # remove old segments; will update _size
        self.removeSegment(segment1)
        self.removeSegment(segment2)
        #assert (line1, left1) not in self._dct
        #assert (line1, right1) not in self._dct
        #assert (line1, left2) not in self._dct
        #assert (line1, right2) not in self._dct
        # add a new merged one; need to update _size by hand
        mergedStats = tuple(self._segmStatsMerges[i](stats1[i], stats2[i]) for i in range(len(self._segmStats)))
        self._dct[(line1, left1)] = (left1, right2, mergedStats)  #linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        self._dct[(line1, right2)] = (left1, right2, mergedStats)  #linearSum1 + linearSum2, squaresSum1 + squaresSum2)
        #print(segment1, segment2, "->", (line1, left1, right2))
        self._size += 1
        self._line_segms[line1] += 1
        return (line1, left1, right2)
            
    def getSegments(self):
        res = []
        for (line, leftIdx) in self._dct.keys():
            begin, end, _ = self._dct[(line, leftIdx)]
            res.append((line, begin, end))
        return res
        
    def getSegmentLeft(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, right = segment
        if (line, left - 1) not in self._dct:
            return None
        segmLeft, segmRight, _ = self._dct[(line, left - 1)]
        #print(left, right, "!", segmLeft, segmRight, left - 1)
        #assert segmRight == left - 1
        #assert self._dct[(line, segmLeft)][0] == segmLeft and self._dct[(line, segmLeft)][1] == segmRight
        return (line, segmLeft, segmRight)
    
    def getSegmentRight(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, right = segment
        if (line, right + 1) not in self._dct:
            return None
        segmLeft, segmRight, _ = self._dct[(line, right + 1)]
        #assert segmLeft == right + 1
        #assert self._dct[(line, segmRight)][0] == segmLeft and self._dct[(line, segmRight)][1] == segmRight
        return (line, segmLeft, segmRight)
    
    def getSegmentStats(self, segment):
        if not self.segmentInDict(segment):
            return None
        line, left, _ = segment
        _, _, stats = self._dct[(line, left)]
        #print("!", self._dct[(line, left)])
        return stats #(linearSum, squaresSum)

    @staticmethod
    def getFinalSegments(merges, shape, padMask=None):  # shape needs to be B x W, without height!

        visited = np.zeros(shape, dtype=np.int32)
        finalSegments = []
        for i in range(len(merges)-1,-1,-1):
            leftSegm, rightSegm = merges[i]
            line, beginLeft, endLeft = leftSegm
            if visited[line][beginLeft] != 0:
                continue  # merge already seen
            _, beginRight, endRight = rightSegm
            finalSegments.append((line, beginLeft, endRight))
            visited[line][beginLeft:(endRight+1)] = 1

        # add length-1 segments that are there not padded but were not a part of any merge
        for i in range(visited.shape[0]):
            for j in range(visited.shape[1]):
                if not visited[i][j] and (padMask is None or not padMask[i][j]):
                    finalSegments.append((i, j, j))

        lineCounter = 0
        prevLine = 0  # don't append useless 0 at the beginning
        res = {}  # {(line, #ofSegmentInLine): (line, beginIdx, endIdx)}
        segmentsInLines = []  # numbers of segments in lines
        for line, begin, end in sorted(finalSegments):
            if line != prevLine:
                prevLine = line
                segmentsInLines.append(lineCounter)
                lineCounter = 0
            res[(line, lineCounter)] = (line, begin, end)
            lineCounter += 1
        segmentsInLines.append(lineCounter)

        return res, segmentsInLines  # there will be always at least 1 segment in a line