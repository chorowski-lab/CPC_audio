
# because why allow for a prefix of returned values to be tensors and allow dicts, lists etc in suffix and just don't add them to graph
# when you can make everyone have to convert things to and from tensors without real reason

# also in DataParallel ofc it's better to demand every GPU has to return exact same shape and not only on dim=0
# to make it even harder to encode and return a dict
# people who think there is only one simplest usecase and make things only work in this case 
# (when there is no reason not to work in other ones and rather not much work to do (or even literally none) for that), shouldn't write libraries

import torch

def convertNumToTens(num):
    return torch.tensor([num]).cpu()

def convertNumTensBack(numTens):
    return numTens[0].item()

def convert3ValueIntSetToInt32Tens(s):  # TODO try with cython
    tens = torch.zeros((3*len(s),), dtype=torch.int32).cpu().numpy()
    i = 0
    for a, b, c in s:
        tens[i] = a
        tens[i+1] = b
        tens[i+2] = c
        i += 3
    return torch.tensor(tens)

def padTens3ValueSetToLength(t, l):  # l is number of 3-num entries
    #print(t.shape[0], l)
    assert t.shape[0] <= l*3
    tens = torch.zeros((3*l,), dtype=torch.int32).cuda()
    tens[:t.shape[0]] = t.cuda()
    tens[t.shape[0]:] = -1
    return tens.cpu()

# this assumes valid values are > 0
def convertTens3ValueSetBack(setTens):  # this is only for slow metric, can leave it as it is
    s = set()
    setTens = setTens.cpu().numpy()
    i = 0
    while i < setTens.shape[0]:
        if setTens[i].item() < 0:  # needed to make encoded dicts on every GPU same length because torch is as user unfriendly as it can
            break
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
    tens = tens.cpu().numpy()
    padMask = padMask.cpu().numpy()
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
    sconv2 = padTens3ValueSetToLength(sconv, 5)
    sback2 = convertTens3ValueSetBack(sconv2)
    print(s, sback, sconv)
    print(s, sback2, sconv2)

    l = [[3,5,8], [2,10], [1,3,5,4]]
    lconv = convert2DimListsToInt32TensorAndMask(l)
    lback = convert2DimListTensBack(lconv)
    print(l, lback, lconv)

