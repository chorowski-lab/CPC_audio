
# because why allow for a prefix of returned values to be tensors and allow dicts, lists etc in suffix and just don't add them to graph
# when you can make everyone have to convert things to and from tensors without real reason

import torch

def convertNumToTens(num):
    return torch.tensor([num]).cpu()

def convertNumTensBack(numTens):
    return numTens[0].item()

def convert3ValueIntSetToInt32Tens(s):
    tens = torch.zeros((3*len(s),), dtype=torch.int32).cpu()
    i = 0
    for a, b, c in s:
        tens[i] = a
        tens[i+1] = b
        tens[i+2] = c
        i += 3
    return tens

def convertTens3ValueSetBack(setTens):
    s = set()
    i = 0
    while i < setTens.shape[0]:
        s.add((setTens[i].item(), setTens[i+1].item(), setTens[i+2].item()))
        i += 3
    return s

# could also make a variant for any nested list, but didn't need that
def convert2DimListsToInt32TensorAndMask(l):
    maxLen = max(map(len, l))
    tens = torch.zeros(len(l), maxLen, dtype=torch.int32).cpu()
    padMask = torch.ones(len(l), maxLen, dtype=torch.int32).cpu()
    for i, subl in enumerate(l):
        padMask[i, :len(subl)] = 0
        tens[i, :len(subl)] = torch.tensor(subl, dtype=torch.int32).cpu()
    return (tens, padMask)

def convert2DimListTensBack(tup):
    tens, padMask = tup
    l = []
    assert tens.shape == padMask.shape
    for i in range(tens.shape[0]):
        l.append([])
        for j in range(tens.shape[1]):
            if padMask[i, j] == 1:
                break
            l[-1].append(tens[i, j].item())
    return l



if __name__ == "__main__":

    x = 1
    xconv = convertNumToTens(x)
    xback = convertNumTensBack(xconv)
    print(x, xback, xconv)

    s = set([(1,2,3), (7,8,3), (0,0,2)])
    sconv = convert3ValueIntSetToInt32Tens(s)
    sback = convertTens3ValueSetBack(sconv)
    print(s, sback, sconv)

    l = [[3,5,8], [2,10], [1,3,5,4]]
    lconv = convert2DimListsToInt32TensorAndMask(l)
    lback = convert2DimListTensBack(lconv)
    print(l, lback, lconv)

