# distutils: language = c++


# cythonize -a -i costsegmcython.pyx

from cython cimport floating
import numpy as np
cimport numpy as np
#import _heapq
#import heapq
from libcpp.queue cimport priority_queue
from libcpp.set cimport set as cset
from libcpp.pair cimport pair
from copy import deepcopy

# cdef struct tuple7:
#     float x1
#     int x2
#     int x3
#     int x4
#     int x5
#     int x6
#     int x7

# https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html

DTYPE_NP_INT = np.int

ctypedef np.int_t DTYPE_NP_INT_T

def fast_thing(floating[:, :] f):

    cdef int h = f.shape[0]
    cdef int w = f.shape[1]
    cdef double el = 0.

    cdef floating[:, :] m = np.zeros([h, w], dtype=np.double)

    cdef priority_queue[double] pq

    #pq = []

    for i in range(h):
        for j in range(w):
            m[i,j] = m[i,j] + f[i,j]
            #el = f[i,j]
            #_heapq.heappush(pq, el)
            pq.push(f[i,j])

    cdef double minel = pq.top()  #_heapq.heappop(pq)

    return m, minel

def costSegm(costs, shape, maxCost, k, minSegmsInLine):
    costs = costs.cpu().numpy()
    if maxCost is None:
        maxCost = -1
    return costSegmFast(costs, shape[0], shape[1], maxCost, k, minSegmsInLine)

cdef bint tup7comp(tuple7 x, tuple7 y):
    return x.x1 < y.x1

# cdef bint tup3comp(tuple3 x, tuple3 y):
#     if (x.x1 < y.x1):
#         return True
#     if (x.x2 < y.x2):
#         return True
#     if (x.x3 < y.x3):
#         return True

# https://stackoverflow.com/questions/59463921/a-priority-queue-with-a-custom-comparator-in-cython
cdef extern from "cpp_pq.hpp":
    cdef cppclass cpp_pq:
        cpp_pq(...) except +
        void push(tuple7)
        tuple7 top()
        void pop()
        bint empty()
        int size()

    cdef struct tuple7:
        float x1
        int x2
        int x3
        int x4
        int x5
        int x6
        int x7

    # cdef cppclass cpp_set:
    #     # iterator stuff copied from original libcpp set which doesn't allow for putting own structs
    #     cppclass iterator:
    #         tuple3& operator*()
    #         iterator operator++()
    #         iterator operator--()
    #         bint operator==(iterator)
    #         bint operator!=(iterator)
    #     cpp_set(...) except +
    #     void insert(tuple3)
    #     void erase(tuple3)
    #     iterator find()
    #     iterator begin()
    #     iterator end()
    #     int size()

# cdef struct tuple3:
#     int x1
#     int x2
#     int x3

def costSegmFast(floating[:, :, :] costs, int h0, int w0, double maxCost, int k, int minSegmsInLine):
    #costs = costs.cpu().numpy()
    maxSegmLen = costs.shape[0]
    #print(f"maxsegmlen: {maxSegmLen}")
    # shape: B x N
    #h, w = shape[0], shape[1]
    #print("!!", h, w)
    cdef int h = h0
    cdef int w = w0
    #print("!!", h, w)
    cdef cset[pair[pair[int,int],int]] segms = set()  #cpp_set(tup3comp) #set()  #{}
    #cdef cpp_set segms = cpp_set(tup3comp)
    cdef np.ndarray[DTYPE_NP_INT_T, ndim=2] lenOnRight = np.zeros([h, w], dtype=DTYPE_NP_INT)  #{}
    cdef np.ndarray[DTYPE_NP_INT_T, ndim=2] lenOnLeft = np.zeros([h, w], dtype=DTYPE_NP_INT)  #{}
    #cdef tuple3 tup3
    cdef pair[pair[int,int],int] tup3
    for iii in range(h):
        for jjj in range(w):
            #print(i,j)
            tup3.first.first = iii
            tup3.first.second = jjj
            tup3.second = 1
            segms.insert(tup3)  #((iii,jjj,1)) #= costs[0,i,j]
            lenOnRight[iii,jjj] = 1  #[(iii,jjj)] = 1
            lenOnLeft[iii,jjj] = 1 #[(iii,jjj)] = 1
    #cdef priority_queue[tuple7] pq = cpp_pq(tup7comp)  #priority_queue[tuple7](tup7comp)
    cdef cpp_pq pq = cpp_pq(tup7comp)
    cdef tuple7 segmData
    for iii in range(h):
        for jjj in range(w-1):
            
            segmData.x1 = -(costs[1,iii,jjj+1])
            segmData.x2 = iii
            segmData.x3 = jjj
            segmData.x4 = 1
            segmData.x5 = iii
            segmData.x6 = jjj + 1
            segmData.x7 = 1
            #heappush(pq, (costs[1,iii,jjj+1].item(),iii,jjj,1,iii,jjj+1,1))
            pq.push(segmData)
    ##linesSegms = [w for _ in range(h)]
    cdef np.ndarray[DTYPE_NP_INT_T, ndim=1] linesSegms = np.ones([h], dtype=DTYPE_NP_INT) * w
    #numSegms = len(segms)
    cdef int loopIters = 0
    #print("--", len(pq))
    cdef float costNow = 0
    segmsWhenCost = None
    segmsWhenK = None
    cdef bint costPresent = True
    if maxCost is None or maxCost < 0:
        maxCost = -1
        costPresent = False
    cdef tuple7 mergeData
    cdef int i1,j1,l1,i2,j2,l2
    cdef float cost
    cdef pair[pair[int,int],int] segm1, segm2, newSegm
    while pq.size() > 0 and (segms.size() > k or costNow <= maxCost):  #numSegms > k:
        loopIters += 1
        ###cost, i1, j1, l1, i2, j2, l2 = heappop(pq)
        mergeData = pq.top()
        pq.pop()
        cost = -(mergeData.x1)
        i1 = mergeData.x2
        j1 = mergeData.x3
        l1 = mergeData.x4
        i2 = mergeData.x5
        j2 = mergeData.x6
        l2 = mergeData.x7

        segm1.first.first = i1
        segm1.first.second = j1
        segm1.second = l1
        segm2.first.first = i2
        segm2.first.second = j2
        segm2.second = l2

        #print(":1", i1, j1, l1, i2, j2, l2)
        #print(segms, segms.find(segm1) == segms.end())
        if (segms.find(segm1) == segms.end()) or (segms.find(segm2) == segms.end()) or linesSegms[i1] <= minSegmsInLine:
        #(i1,j1,l1) not in segms or (i2,j2,l2) not in segms or linesSegms[i1] <= minSegmsInLine:
            continue
        #print(":", i1, j1, l1, i2, j2, l2)
        
        ###print(";", cost)

        # before the merge that could have too big cost
        if costPresent and cost > maxCost and segmsWhenCost is None:
            ###print("A")
            segmsWhenCost = set()  #deepcopy(segms)
            for x in segms:
                segmsWhenCost.add((x.first.first, x.first.second, x.second))
            costWhenCost = costNow
            kWhenCost = segms.size()

        costNow = cost

        segms.erase(segm1) #((i1,j1,l1))
        segms.erase(segm2) #((i2,j2,l2))
        newLen = l1+l2
        newSegm.first.first = i1
        newSegm.first.second = j2
        newSegm.second = l1+l2
        segms.insert(newSegm)  #((i1, j2, l1+l2))  # = costs[l1+l2-1, i1, j2]
        linesSegms[i1] = linesSegms[i1] - 1
        lenOnRight[(i1,j1-l1+1)] = newLen
        lenOnLeft[(i1,j2)] = newLen
        #numSegms -= 1
        ###print("@", newLen, i1, j2-newLen+1, j2, "|", j1-l1+1, j2, "&", i1, j1, l1, i2, j2, l2)
        if j1-l1 >= 0:
            ll = lenOnLeft[i1,j1-l1]  #[(i1,j1-l1)]
            ###print("ll", ll)
            if newLen+ll <= maxSegmLen:
                #cdef tuple7 segmData
                segmData.x1 = -(costs[newLen+ll-1,i1,j2])
                segmData.x2 = i1
                segmData.x3 = j1-l1
                segmData.x4 = ll
                segmData.x5 = i1
                segmData.x6 = j2
                segmData.x7 = newLen
                #heappush(pq, (costs[newLen+ll-1,i1,j2].item(),i1,j1-l1,ll,i1,j2,newLen))
                pq.push(segmData)
                ###print("ADD", newLen+ll-1, i1, j2, "|", (costs[newLen+ll-1,i1,j2].item(),i1,j1-l1,ll,i1,j2,newLen))
                #print((i1,j1-l1,ll) in segms, (i1,j2,newLen) in segms)
        if j2+1 < w:
            lr = lenOnRight[i1, j2+1]  #[(i1, j2+1)]
            ###print("lr", lr)
            if newLen+lr <= maxSegmLen:
                #cdef tuple7 segmData
                segmData.x1 = -(costs[newLen+lr-1,i1,j2+lr])
                segmData.x2 = i1
                segmData.x3 = j2
                segmData.x4 = newLen
                segmData.x5 = i1
                segmData.x6 = j2+lr
                segmData.x7 = lr
                #heappush(pq, (costs[newLen+lr-1,i1,j2+lr].item(),i1,j2,newLen,i1,j2+lr,lr))
                pq.push(segmData)
                ###print("ADD", newLen+lr-1,i1,j2+lr, "|", (costs[newLen+lr-1,i1,j2+lr].item(),i1,j2,newLen,i1,j2+lr,lr))
                #print((i1,j2,newLen) in segms, (i1,j2+lr,lr) in segms)
        #print(len(pq), len(segms))
        #print(segms)

        if segms.size() == k:
            ###print("B")
            segmsWhenK = set()  #deepcopy(segms)
            for x in segms:
                segmsWhenK.add((x.first.first, x.first.second, x.second))
            costWhenK = costNow
            kWhenK = k
        

    if costPresent and segmsWhenCost is None:
        ###print("C")
        segmsWhenCost = set()  #deepcopy(segms)
        for x in segms:
            segmsWhenCost.add((x.first.first, x.first.second, x.second))
        costWhenCost = costNow
        kWhenCost = int(segms.size())

    # minsegmsinline reached
    if segmsWhenK is None:
        ###print("D")
        segmsWhenK = set()  #deepcopy(segms)
        for x in segms:
            segmsWhenK.add((x.first.first, x.first.second, x.second))
        costWhenK = costNow
        kWhenK = int(segms.size())

    #print(len(pq), len(segms))

    #print(f"Loop iters: {loopIters}")
    if costPresent:
        return (segmsWhenCost, costWhenCost, kWhenCost), (segmsWhenK, costWhenK, kWhenK)  #segms
    else:
        return None, (segmsWhenK, costWhenK, kWhenK)
