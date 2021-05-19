

from cpc.model import CPCModel
from copy import deepcopy
import torch

class EncFake:
    def __call__(self, x):
        return x.permute(0,2,1)
    def getDimOutput(self):
        return 2

encFake = EncFake()  #lambda x: x.permute(0,2,1)
ARFake = lambda x: x
fcmSettingsBase = {
    "FCMproject": False,
    "numProtos": 3, 
    "mBeforeAR": None, 
    "leftProtos": None,
    "pushDegFeatureBeforeAR": None, 
    "mAfterAR": None,
    "pushDegCtxAfterAR": None,
    "pushDegAllAfterAR": None,
    "reprsConcat": False, #,
    "reprsConcatNormSumsNotLengths": False,
    "pushLossWeightEnc": None,
    "pushLossWeightCtx": None,
    "VQpushEncCenterWeightOnTopConv": None,
    "VQpushEncCenterWeightOnlyAR": None,
    "VQpushEncCenterWeightOnlyCriterion": None,
    "VQgradualStart": None,
    "VQpushCtxCenterWeight": None,
    "pushLossLinear": None,
    "pushLossGradual": False,
    "pushLossProtosMult": None,
    "pushLossCenterNorm": False,
    "pushLossPointNorm": False,
    "pushLossNormReweight": False,
    "hierARshorten": None,
    "hierARmergePrior": None
    #"reprsConcatDontIncreaseARdim": args.FCMreprsConcatIncreaseARdim
}
points = torch.tensor([[[1.,1.], [2.,2.]], [[3.,3.], [4.,4.]]])
centers = torch.tensor([[1.7,1.7], [2.2,2.2]])
label = None

fcmSettings1 = deepcopy(fcmSettingsBase)
fcmSettings1["VQpushEncCenterWeightOnTopConv"] = 0.1

CPCmodel1 = CPCModel(encFake, ARFake, fcmSettings1)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel1(points, label, None, None, centers, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---------------------after1")

fcmSettings2 = deepcopy(fcmSettingsBase)
fcmSettings2["VQpushEncCenterWeightOnTopConv"] = 0.1
fcmSettings2["VQgradualStart"] = 1

CPCmodel2 = CPCModel(encFake, ARFake, fcmSettings2)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel2(points, label, None, None, centers, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---------------------after2")

fcmSettings3 = deepcopy(fcmSettingsBase)
fcmSettings3["VQpushEncCenterWeightOnlyAR"] = 0.1

CPCmodel3 = CPCModel(encFake, ARFake, fcmSettings3)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel3(points, label, None, None, centers, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---------------------after3")

fcmSettings4 = deepcopy(fcmSettingsBase)
fcmSettings4["VQpushEncCenterWeightOnlyCriterion"] = 0.1

CPCmodel4 = CPCModel(encFake, ARFake, fcmSettings4)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel4(points, label, None, None, centers, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---")
pushLoss, closestCountsDataPar, c_feature, encoded_data = \
    CPCmodel4(c_feature, encoded_data, c_feature, encoded_data, centers, 
              (3,5), True, False)
print(encoded_data)
print(c_feature)
print("---------------------after4")

fcmSettings5 = deepcopy(fcmSettingsBase)
fcmSettings5["VQpushEncCenterWeightOnlyCriterion"] = 0.1
fcmSettings5["VQpushCtxCenterWeight"] = 0.1

CPCmodel5 = CPCModel(encFake, ARFake, fcmSettings5)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel5(points, label, None, None, centers, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---")
pushLoss, closestCountsDataPar, c_feature, encoded_data = \
    CPCmodel5(c_feature, encoded_data, c_feature, encoded_data, centers, 
              (3,5), True, False)
print(encoded_data)
print(c_feature)
print("---------------------after5")

points2 = torch.tensor([[[1.,1.], [2.,2.4]], [[3.,3.], [4.,4.8]]])
centers2 = torch.tensor([[1.7,1.7], [1.,1.3]])
fcmSettings6 = deepcopy(fcmSettingsBase)
fcmSettings6["VQpushEncCenterWeightOnlyCriterion"] = 1
fcmSettings6["VQpushCtxCenterWeight"] = 0.5
fcmSettings6["pushLossCenterNorm"] = True
fcmSettings6["pushLossPointNorm"] = True

CPCmodel6 = CPCModel(encFake, ARFake, fcmSettings6)
c_feature, encoded_data, pure_enc, label, pushLoss, segmDictTens = \
    CPCmodel6(points2, label, None, None, centers2, (3,5), False, False)

print(encoded_data)
print(pure_enc)
print(c_feature)
print("---")
pushLoss, closestCountsDataPar, c_feature, encoded_data = \
    CPCmodel6(c_feature, encoded_data, c_feature, encoded_data, centers2, 
              (3,5), True, False)
# this one is tricky! whould push last points in lines in cosine way,
# from y/x = 1.2 to: 1.3 (enc, 1 center weight), 1.25 (ctx, 0.5 center weight)
# as the closest center has 1.3 ratio
print(encoded_data)
print(c_feature)
print("---------------------after6")