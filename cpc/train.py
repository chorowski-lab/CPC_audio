# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import os
import numpy as np
import torch
import time
from copy import deepcopy
import random
import psutil
import sys
#import torchaudio

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.model as model
import cpc.center_model as center_model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
import cpc.eval.linear_separability as linsep
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
import cpc.stats.stat_utils as statutil
from cpc.segm.hier_fast import mergeSlowStats
from cpc.segm.segment_cost_model import SegmentCostModel


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
        else:
            sizeInputSeq = (args.sizeWindow // downsampling)

            # TODO this part can be simplified with m's and reprsConcat
            if args.FCMproject and args.FCMmBeforeAR:

                if not args.FCMreprsConcat:
                    # this could be replaced with encoder.getOutDim, but didn't want to change signature
                    encoderOutDimForCriterion = args.FCMprotosForCriterion  # .FCMprotos
                    # [!] args.hiddenGar already updated with possible FCM dim stuff in main train script
                    #     it is the actual AR dimension - so only needs to be changed here in case of FCMmAfterAR
                    #     changes the dimension after AR
                    ARoutDimForCriterion = args.FCMprotosForCriterion  #args.hiddenGar  # but actually should also be == args.FCMprotos
                else:
                    encoderOutDimForCriterion = args.hiddenEncoder + args.FCMprotosForCriterion
                    ARoutDimForCriterion = args.hiddenEncoder + args.FCMprotosForCriterion

            elif args.FCMproject and args.FCMmAfterAR:

                if not args.FCMreprsConcat:
                    encoderOutDimForCriterion = args.FCMprotosForCriterion
                    ARoutDimForCriterion = args.FCMprotosForCriterion
                else:
                    encoderOutDimForCriterion = args.hiddenEncoder + args.FCMprotosForCriterion
                    ARoutDimForCriterion = args.hiddenEncoder + args.FCMprotosForCriterion

            else:
                encoderOutDimForCriterion = args.hiddenEncoder
                ARoutDimForCriterion = args.hiddenGar
                
            if args.CPCCTC:
                cpcCriterion = sa.CPCUnsupersivedCriterion(args.nPredicts,
                                                        args.CPCCTCNumMatched,
                                                        ARoutDimForCriterion,
                                                        encoderOutDimForCriterion,
                                                        args.negativeSamplingExt,
                                                        allowed_skips_beg=args.CPCCTCSkipBeg,
                                                        allowed_skips_end=args.CPCCTCSkipEnd,
                                                        predict_self_loop=args.CPCCTCSelfLoop,
                                                        learn_blank=args.CPCCTCLearnBlank,
                                                        normalize_enc=args.CPCCTCNormalizeEncs,
                                                        normalize_preds=args.CPCCTCNormalizePreds,
                                                        masq_rules=args.CPCCTCMasq,
                                                        loss_temp=args.CPCCTCLossTemp,
                                                        no_negs_in_match_window=args.CPCCTCNoNegsMatchWin,
                                                        limit_negs_in_batch=args.limitNegsInBatch,
                                                        mode=args.cpc_mode,
                                                        rnnMode=args.rnnMode,
                                                        dropout=args.dropout,
                                                        nSpeakers=nSpeakers,
                                                        speakerEmbedding=args.speakerEmbedding,
                                                        sizeInputSeq=sizeInputSeq)

            else:
                cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                        ARoutDimForCriterion,
                                                        encoderOutDimForCriterion,
                                                        args.negativeSamplingExt,
                                                        mode=args.cpc_mode,
                                                        rnnMode=args.rnnMode,
                                                        dropout=args.dropout,
                                                        nSpeakers=nSpeakers,
                                                        speakerEmbedding=args.speakerEmbedding,
                                                        sizeInputSeq=sizeInputSeq, 
                                                        modelLengthInAR=args.modelLengthInAR)
    elif args.pathPhone is not None:
        if not args.CTC:
            cpcCriterion = cr.PhoneCriterion(dimFeatures,
                                             nPhones, args.onEncoder,
                                             nLayers=args.nLevelsPhone)
        else:
            cpcCriterion = cr.CTCPhoneCriterion(dimFeatures,
                                                nPhones, args.onEncoder)
    else:
        cpcCriterion = cr.SpeakerCriterion(dimFeatures, nSpeakers)
    return cpcCriterion


def loadCriterion(pathCheckpoint, downsampling, nSpeakers, nPhones):
    _, _, locArgs, _ = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def trainStep(dataLoader,
              cpcModel,
              centerModel,
              segmentCostModel,
              cpcCriterion,
              optimizer,
              scheduler,
              loggingStep,
              epochNrs):

    cpcModel.train()
    cpcCriterion.train()

    start_time = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iter = 0

    epochNr, totalEpochs = epochNrs
    normalBatchSize = 0

    for step, fulldata in enumerate(dataLoader):
        batchData, labelData = fulldata
        normalBatchSize = max(normalBatchSize, batchData.shape[0])  # if weird small one goes first, some stats can get a bit spoiled but not much
        #%#print("::", normalBatchSize)
        label = labelData['speaker']
        labelPhone = labelData['phone']
        #print("!!!", labelData.keys())
        n_examples += batchData.size(0)
        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        labelPhone = labelPhone.cuda(non_blocking=True)

        #print("!!!", batchData.shape)
        # [!] ok, this has concatenated shape and later DataParallel splits it by 2
        #     so can do this 2-stage forward as I did

        
        givenCenters = centerModel.centersForStuff(epochNr) if centerModel is not None else None
        numGPUs = len(cpcModel.device_ids)
        if givenCenters is not None:
            givenCenters = givenCenters.repeat(numGPUs,1)
        #print(dir(cpcModel))
        #print(1/0)

        # https://discuss.pytorch.org/t/dataparallel-only-supports-tensor-output/34519
        # did it like that so that code in other places doesn;t need to be changed
        # also, can't just check cpcModel.hasPushLoss as dataParallel makes it harder to access
        if centerModel is not None:
            centerModel.inputsBatchUpdate(batchData, epochNrs, cpcModel)
        if segmentCostModel is not None:
            maxAllowedSegmCost = segmentCostModel.getCurrentMaxCostEstimator()
        else:
            maxAllowedSegmCost = None
        c_feature, encoded_data, pure_enc, label, labelPhoneByGPU, pushLoss, segmSetTens, batchAimCostSegmTens, batchActualKTens = cpcModel(batchData, label, labelPhone, maxAllowedSegmCost, givenCenters, epochNrs, False, False)
        if segmentCostModel is not None and batchData.shape[0] == normalBatchSize:  # avoiding updating with smaller batches as would spoil average segm number stats
            segmentCostModel.batchUpdate(batchAimCostSegmTens, batchActualKTens)
        #print("!!!!", label.shape)
        if centerModel is not None:
            centerUpdateRes = centerModel.encodingsBatchUpdate(encoded_data, epochNrs, cpcModel, label=labelPhone)
            #print(f"!!! centerUpdate is None: {centerUpdateRes is None}")
            if centerUpdateRes is None:
                centerUpdateRes = {}
            DM = centerModel.getDM(epochNr)
        else:
            centerUpdateRes = {}
            DM = None
        # [!] baseEncDim returned (in tensor) if push loss

        if pushLoss is not None:  # else
            baseEncDim = pushLoss[0].item()
            c_feature1 = c_feature.clone()
            #encoded_data1 = encoded_data.clone()
            c_feature2 = c_feature.clone()
            encoded_data2 = pure_enc #encoded_data.clone()
            c_feature = c_feature1
            #encoded_data = encoded_data1
            pushLoss, closestCountsDataPar, c_feature, encoded_data = \
                cpcModel(c_feature, encoded_data, c_feature2, encoded_data2, givenCenters, epochNrs, True, False)
            closestCounts = closestCountsDataPar.sum(dim=0).view(-1)
            c_feature.retain_grad()  # grad can be retained after possible VQ-VAE, as grad is copied anyway with detach-some-stuff trick
            c_feature2.retain_grad()
            encoded_data.retain_grad()
            encoded_data2.retain_grad()
            
        allCriterionLosses, allAcc, _ = cpcCriterion(c_feature, encoded_data, label, None)

        #totLoss = allCriterionLosses.sum()   # a bit below in if-else now
        #allCriterionLosses.retain_grad()
        if pushLoss is not None:
            # pushLoss will have shape depeding on dataParallel
            #print(allCriterionLosses.shape, pushLoss.shape)
            totLoss = allCriterionLosses.sum() + pushLoss.sum()  # pushLoss is an average per-vector
            
        else:
            totLoss = allCriterionLosses.sum()


        totLoss.backward()

        if pushLoss is not None:
            nonpushgradenc = encoded_data.grad[:, :, :baseEncDim].abs().mean().detach()
            if encoded_data2.grad is not None:
                pushgradenc = encoded_data2.grad.abs().mean().detach()
            else:
                pushgradenc = torch.zeros(1)
            nonpushgradctx = c_feature.grad[:, :, :baseEncDim].abs().mean().detach()
            if c_feature2.grad is not None:
                pushgradctx = c_feature2.grad.abs().mean().detach()
            else:
                pushgradctx = torch.zeros(1)

        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["phones_train"] = np.zeros(labelData['phoneNr'][0].item())
            logs["locLoss_train"] = np.zeros(allCriterionLosses.size(1))
            logs["locAcc_train"] = np.zeros(allCriterionLosses.size(1))
            if pushLoss is not None:
                logs["grad_enc_cpc_train"] = np.zeros(1)
                logs["grad_enc_push_train"] = np.zeros(1)
                logs["grad_ctx_cpc_train"] = np.zeros(1)
                logs["grad_ctx_push_train"] = np.zeros(1)
        if "labelCounts" not in logs and "labelCounts" in centerUpdateRes:
            logs["labelCounts"] = np.zeros((1,1))
        if "centersDM" not in logs and DM is not None:
            logs["centersDM"] = np.zeros((1,1))
        if "pushloss_closest" not in logs and pushLoss is not None:
            logs["pushloss_closest"] = np.zeros(closestCounts.shape[0])
        if "merge_stats_train" not in logs and segmSetTens is not None:
            logs["merge_stats_train"] = np.zeros((1,1))

        iter += 1
        logs["locLoss_train"] += (allCriterionLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()
        phoneIndices, phoneIndicesCounts = torch.unique(labelPhone.detach(), return_counts=True)
        phoneCounts = torch.zeros(logs["phones_train"].shape, dtype=torch.float32).cuda()
        phoneCounts[phoneIndices] += phoneIndicesCounts
        logs["phones_train"] = logs["phones_train"] + phoneCounts.cpu().numpy()
        if pushLoss is not None:
            # already detached previously
            logs["grad_enc_cpc_train"] += nonpushgradenc.cpu().numpy()
            logs["grad_enc_push_train"] += pushgradenc.cpu().numpy()
            logs["grad_ctx_cpc_train"] += nonpushgradctx.cpu().numpy()
            logs["grad_ctx_push_train"] += pushgradctx.cpu().numpy()
            #print("!", logs["pushloss_closest"].shape, closestCounts.shape)
            logs["pushloss_closest"] += closestCounts.detach().cpu().numpy()
        if "labelCounts" in centerUpdateRes:
            logs["labelCounts"] = centerUpdateRes["labelCounts"].detach().cpu().numpy() + logs["labelCounts"]
        if DM is not None:
            logs["centersDM"] = DM.detach().cpu().numpy() + logs["centersDM"]
        if segmSetTens is not None and epochNr == totalEpochs:  # this stat is super slow, only do at the very end
            numPhones = labelData['phoneNr'][0].item()
            #--print(f"-------->*************** train numPhones: {numPhones}, label shape {labelPhone.shape}")
            #%#print(f"----> SHAPE [0] OF SET TENSOR train: {segmSetTens.shape[0]}")
            for i in range(segmSetTens.shape[0]):
                mergesNums, _ = mergeSlowStats(segmSetTens[i], labelPhoneByGPU[i], numPhones)
                logs["merge_stats_train"] = mergesNums + logs["merge_stats_train"]

        if (step + 1) % loggingStep == 0:
            new_time = time.perf_counter()
            elapsed = new_time - start_time
            print(f"Update {step + 1}")
            print(f"elapsed: {elapsed:.1f} s")
            print(
                f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
            locLogs = utils.update_logs(logs, loggingStep, lastlogs)
            lastlogs = deepcopy(logs)
            utils.show_logs("Training loss", locLogs)
            if segmentCostModel is not None:
                segmentCostModel.showCurrentStats()
            start_time, n_examples = new_time, 0

    if centerModel is not None:
        centerModel.epochUpdate(epochNrs, cpcModel)
        centerModel.printLens()

    if scheduler is not None:
        scheduler.step()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Average training loss on epoch", logs)
    if segmentCostModel is not None:
        segmentCostModel.showCurrentStats()
    return logs


def valStep(dataLoader,
            cpcModel,
            centerModel,
            segmentCostModel,
            cpcCriterion,
            epochNrs):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    epochNr, totalEpochs = epochNrs

    for step, fulldata in enumerate(dataLoader):

        batchData, labelData = fulldata
        label = labelData['speaker']
        labelPhone = labelData['phone']

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with torch.no_grad():
            givenCenters = centerModel.centersForStuff(epochNr) if centerModel is not None else None
            numGPUs = len(cpcModel.device_ids)
            if givenCenters is not None:
                givenCenters = givenCenters.repeat(numGPUs,1)
            if segmentCostModel is not None:
                maxAllowedSegmCost = segmentCostModel.getCurrentMaxCostEstimator()
            else:
                maxAllowedSegmCost = None
            c_feature, encoded_data, pure_enc, label, labelPhoneByGPU, pushLoss, segmSetTens, _, _ = cpcModel(batchData, label, labelPhone, maxAllowedSegmCost, givenCenters, epochNrs, False, False)
            allLosses, allAcc, _ = cpcCriterion(c_feature, encoded_data, label, None)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))
        if "merge_stats_val" not in logs and segmSetTens is not None:
            logs["merge_stats_val"] = np.zeros((1,1))

        iter += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()
        if segmSetTens is not None and epochNr == totalEpochs:  # this stat is super slow, only do at the very end
            numPhones = labelData['phoneNr'][0].item()
            #print(f"-------->*************** val numPhones: {numPhones}")
            #%#print(f"----> SHAPE [0] OF SET TENSOR val: {segmSetTens.shape[0]}")
            for i in range(segmSetTens.shape[0]):
                mergesNums, _ = mergeSlowStats(segmSetTens[i], labelPhoneByGPU[i], numPhones)
                logs["merge_stats_val"] = mergesNums + logs["merge_stats_val"]

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs


def captureStep(
            dataLoader,
            cpcModel,
            centerModel,
            segmentCostModel,
            cpcCriterion,
            captureOptions,
            captureStatsCollector,
            epochNrs):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    epochNr, totalEpochs = epochNrs

    capturePath = captureOptions['path']
    whatToSave = captureOptions['what']
    cpcCaptureOpts = []
    if 'pred' in whatToSave:
        cpcCaptureOpts.append('pred')
    if 'cpcctc_align' in whatToSave:
        cpcCaptureOpts.append('cpcctc_align')
    if 'cpcctc_log_scores' in whatToSave:
        cpcCaptureOpts.append('cpcctc_log_scores')

    # they merge (perhaps each speaker's) audio into one long chunk
    # and AFAIU sample can begin in one file and end in other one
    # so won't try to mess up with tracking filenames, saving samples just as 1, 2, etc.

    if captureStatsCollector:
        captureStatsCollector.zeroStats()

    batchBegin = 0
    epochDir = os.path.join(capturePath, str(epochNr))
    if not os.path.exists(epochDir):
        os.makedirs(epochDir)
    for sub in whatToSave:
        if not os.path.exists(os.path.join(epochDir, sub)):
            os.makedirs(os.path.join(epochDir, sub))

    for step, fulldata in enumerate(dataLoader):

        batchData, labelData = fulldata
        labelSpeaker = labelData['speaker']
        batchEnd = batchBegin + batchData.shape[0] - 1

        batchData = batchData.cuda(non_blocking=True)
        labelSpeaker = labelSpeaker.cuda(non_blocking=True)

        with torch.no_grad():

            givenCenters = centerModel.centersForStuff(epochNr) if centerModel is not None else None
            numGPUs = len(cpcModel.device_ids)
            if givenCenters is not None:
                givenCenters = givenCenters.repeat(numGPUs,1)
            if segmentCostModel is not None:
                maxAllowedSegmCost = segmentCostModel.getCurrentMaxCostEstimator()
            else:
                maxAllowedSegmCost = None
            c_feature, encoded_data, pure_enc, labelSpeaker, _, _, _, _, _ = cpcModel(batchData, labelSpeaker, None, maxAllowedSegmCost, givenCenters, epochNrs, False, False)
            allLosses, allAcc, captured = cpcCriterion(c_feature, encoded_data, labelSpeaker, cpcCaptureOpts)
        
            # saving it with IDs like that assumes deterministic order of elements
            # which is there as dataLoader is a sequential one here
            if 'conv_repr' in whatToSave:
                # encoded data shape: batch_size x len x repr_dim
                torch.save(encoded_data.cpu(), os.path.join(epochDir, 'conv_repr', f'conv_repr_batch{batchBegin}-{batchEnd}.pt'))
            if 'ctx_repr' in whatToSave:
                # ctx data shape: also batch_size x len x repr_dim
                torch.save(c_feature.cpu(), os.path.join(epochDir, 'ctx_repr', f'ctx_repr_batch{batchBegin}-{batchEnd}.pt'))
            if 'speaker_align' in whatToSave:
                # speaker data shape: batch_size (1-dim, each one in batch is whole by 1 speaker)
                torch.save(labelSpeaker.cpu(), os.path.join(epochDir, 'speaker_align', f'speaker_align_batch{batchBegin}-{batchEnd}.pt'))
            if 'phone_align' in whatToSave:
                # phone alignment data shape: batch_size x len
                torch.save(labelData['phone'].cpu(), os.path.join(epochDir, 'phone_align', f'phone_align_batch{batchBegin}-{batchEnd}.pt'))
            for cpcCaptureThing in cpcCaptureOpts:
                # pred shape (CPC-CTC): batch_size x (len - num_matched) x repr_dim x num_predicts (or num_predicts +1 if self loop allowed)
                # cpcctc_align shape (CPC-CTC): batch_size x (len - num_matched) x num_matched
                # cpcctc_log_scores shape (CPC-CTC): batch_size x (len - num_matched) x num_matched x num_predicts (or num_predicts +1 if self loop allowed)
                torch.save(captured[cpcCaptureThing].cpu(), os.path.join(epochDir, cpcCaptureThing, 
                            f'{cpcCaptureThing}_batch{batchBegin}-{batchEnd}.pt'))

            if captureStatsCollector:
                allBatchData = {}
                allBatchData['conv_repr'] = encoded_data
                allBatchData['ctx_repr'] = c_feature
                allBatchData['speaker_align'] = labelSpeaker
                if 'phone' in labelData:
                    allBatchData['phone_align'] = labelData['phone']
                # ones below are only ones that need to be captured(saved) in order to be available for stats
                for cpcCaptureThing in captured:
                    allBatchData[cpcCaptureThing] = captured[cpcCaptureThing]

                captureStatsCollector.batchUpdate(allBatchData)

            # TODO maybe later can write that with process pool or something??? but not even sure if makes sense

        batchBegin += batchData.shape[0]

    if captureStatsCollector:
        captureStatsCollector.logStats(epochNr)
    return


def run(trainDataset,
        valDataset,
        captureDatasetWithOptions,
        linsepClassificationTaskConfig,
        batchSize,
        samplingMode,
        cpcModel,
        centerModel,
        segmentCostModel,
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler,
        logs,
        cpcEpochCompleted):

    startEpoch = cpcEpochCompleted + 1  #len(logs["epoch"])
    print(f"Running {nEpoch} epochs, now at {startEpoch}")
    bestAcc = 0
    bestStateDict = None
    start_time = time.time()

    if "epoch" not in logs:
        logs["epoch"] = []
    while len(logs["epoch"]) < startEpoch:
        logs["epoch"].append(None)
    
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    linsepEpochsConfig, linsepFun = linsepClassificationTaskConfig
    assert (captureDataset is None and captureOptions is None) \
        or (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']
    if linsepEpochsConfig is not None:
        linsepEpochsConfig = linsepEpochsConfig.strip()
        if linsepEpochsConfig.startswith('('):
            linsepEpochs = list(map(int, (linsepEpochsConfig[1:-1]).split(',')))
        else:
            eachn = int(linsepEpochsConfig)
            nextok = startEpoch - (startEpoch % eachn)
            if nextok < startEpoch:
                nextok += eachn
            linsepEpochs = list(range(nextok, nEpoch, eachn))
    else:
        linsepEpochs = None
        #print("@@@@@@", eachn, linsepEpochs)

    print(f'DS sizes: train {str(len(trainDataset)) if trainDataset is not None else "-"}, '
        f'val {str(len(valDataset)) if valDataset is not None else "-"}, capture '
        f'{str(len(captureDataset)) if captureDataset is not None else "-"}')

    for epoch in range(startEpoch, nEpoch):

        print(f"Starting epoch {epoch}")
        sys.stdout.flush()
        utils.cpu_stats()

        trainLoader = trainDataset.getDataLoader(batchSize, samplingMode,
                                                True, numWorkers=0)
        
        valLoader = valDataset.getDataLoader(batchSize, 'sequential', False,
                                            numWorkers=0)
        
        if captureDataset is not None and epoch % captureEachEpochs == 0:
            captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                numWorkers=0)
        
        print("Training dataset %d batches, Validation dataset %d batches, batch size %d" %
            (len(trainLoader), len(valLoader), batchSize))

        locLogsTrain = trainStep(trainLoader, cpcModel, centerModel, segmentCostModel, cpcCriterion,
                                optimizer, scheduler, logs["logging_step"], (epoch, nEpoch-1))

        locLogsVal = valStep(valLoader, cpcModel, centerModel, segmentCostModel, cpcCriterion, (epoch, nEpoch-1))

        if captureDataset is not None and epoch % captureEachEpochs == 0:
            print(f"Capturing data for epoch {epoch}")
            captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, captureStatsCollector, (epoch, nEpoch-1))

        currentAccuracy = float(locLogsVal["locAcc_val"].mean())
        if currentAccuracy > bestAcc:
            bestStateDict = deepcopy(fl.get_module(cpcModel).state_dict())  

        locLogsLinsep = {}
        # this performs linsep task for the best CPC model up to date
        if linsepEpochs is not None and epoch != 0 and epoch in linsepEpochs:
            # capturing for current CPC state after this epoch, relying on CPC internal accuracy is vague
            locLogsLinsep = linsepFun(epoch, cpcModel, centerModel, segmentCostModel, (epoch, nEpoch-1))

        print(f'Ran {epoch + 1} epochs '
            f'in {time.time() - start_time:.2f} seconds')

        torch.cuda.empty_cache()

        for key, value in dict(locLogsTrain, **locLogsVal, **locLogsLinsep).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            while len(logs[key]) < len(logs["epoch"]):
                logs[key].append(None)  # for not-every-epoch-logged things
            logs[key].append(value)

        logs["epoch"].append(epoch)

        if pathCheckpoint is not None \
                and (epoch % logs["saveStep"] == 0 or epoch == nEpoch-1):

            modelStateDict = fl.get_module(cpcModel).state_dict()
            criterionStateDict = fl.get_module(cpcCriterion).state_dict()

            fl.save_checkpoint(modelStateDict, criterionStateDict,
                            optimizer.state_dict(), bestStateDict,
                            segmentCostModel, f"{pathCheckpoint}_{epoch}.pt")
            utils.save_logs(logs, pathCheckpoint + "_logs.json")


def onlyCapture(
        captureDatasetWithOptions,
        batchSize,
        cpcModel,
        centerModel,
        segmentCostModel,
        cpcCriterion,
        logs,
        cpcEpochCompleted
):
    startEpoch = cpcEpochCompleted + 1 #len(logs["epoch"])
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    assert (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']
    print(f'Capture DS size: {str(len(captureDataset))}')

    captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                numWorkers=0)
    print(f"Capturing data for model checkpoint after epoch: {startEpoch-1}")
    captureStep(captureLoader, cpcModel, centerModel, segmentCostModel, cpcCriterion, captureOptions, captureStatsCollector, (startEpoch-1, startEpoch-1))


def main(args):

    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7309))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    args = parseArgs(args)

    utils.set_seed(args.random_seed)
    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    loadOptimizer = False
    os.makedirs(args.pathCheckpoint, exist_ok=True)

    cpcEpochCompleted = -1  # enabling to train form checkpoint epoch and not logs epoch, possibly messing up logs but not model
    # needed to move thing below later, as adding some args later
    #json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))

    if args.pathCheckpoint is not None and not args.restart:
        cdata = fl.getCheckpointData(args.pathCheckpoint, noLoadPossible=args.overrideArgsFile)
        if cdata is not None:
            data, logs, locArgs, cpcEpochCompleted = cdata
            print(f"Checkpoint detected at {data}")
            if not args.overrideArgsFile:
                fl.loadArgs(args, locArgs,
                            forbiddenAttr={"nGPU", "pathCheckpoint",
                                        "debug", "restart", "world_size",
                                        "n_nodes", "node_id", "n_gpu_per_node",
                                        "max_size_loaded"})
            args.load, loadOptimizer = [data], True
            args.loadCriterion = True

    logs["logging_step"] = args.logging_step
    logs0 = {"epoch": [], "iter": [], "saveStep": args.save_step}
    for k in logs0:
        if k not in logs:
            logs[k] = logs0[k]

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache)

    if not args.onlyCapture or args.only_classif_metric:
        print(f'Found files: {len(seqNames)} seqs, {len(speakers)} speakers')
        # Datasets
        if args.pathTrain is not None:
            seqTrain = filterSeqs(args.pathTrain, seqNames)
        else:
            seqTrain = seqNames

        if args.pathVal is None:
            random.shuffle(seqTrain)
            sizeTrain = int(0.99 * len(seqTrain))
            seqTrain, seqVal = seqTrain[:sizeTrain], seqTrain[sizeTrain:]
            print(f'Found files: {len(seqTrain)} train, {len(seqVal)} val')
        else:
            seqVal = filterSeqs(args.pathVal, seqNames)

    if args.pathCaptureDS is not None:
        assert args.pathCaptureSave is not None
        whatToSave = []
        if args.captureEverything:
            whatToSave = ['conv_repr', 'ctx_repr', 'speaker_align', 'pred']
            if args.path_phone_data:
                whatToSave.append('phone_align')
            if args.CPCCTC:
                whatToSave.append('cpcctc_align')
                whatToSave.append('cpcctc_log_scores')
        else:
            for argVal, name in zip([args.captureConvRepr, 
                                    args.captureCtxRepr, 
                                    args.captureSpeakerAlign, 
                                    args.capturePhoneAlign,
                                    args.capturePred,
                                    args.captureCPCCTCalign,
                                    args.captureCPCCTClogScores], 
                                    ['conv_repr', 'ctx_repr', 'speaker_align', 'phone_align', 'pred', 'cpcctc_align', 'cpcctc_log_scores']):
                if argVal:
                    whatToSave.append(name)
        ###assert len(whatToSave) > 0
        captureOptions = {
            'path': args.pathCaptureSave,
            'eachEpochs': args.captureEachEpochs,
            'what': whatToSave
        }
        seqCapture = filterSeqs(args.pathCaptureDS, seqNames, 
                                percentage=args.captureDSfreq, totalNum=args.captureDStotNr)
        print(f'Capture files: {len(seqCapture)}')
    else:
        seqCapture = None
        captureOptions = None

    if not args.onlyCapture:
        if args.debug:
            seqTrain = seqTrain[-1000:]
            seqVal = seqVal[-100:]

        phoneLabels, nPhones = None, None
        if args.supervised and args.pathPhone is not None:
            print("Loading the phone labels at " + args.pathPhone)
            phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
            print(f"{nPhones} phones found")
        if args.supervised_classif_metric and args.path_phone_data is not None:
            # so that can use same DS for train & for classif metric
            print("Loading the phone labels at " + args.path_phone_data)
            phoneLabels, nPhones = parseSeqLabels(args.path_phone_data)
            print(f"{nPhones} phones found")

        print("")
        print(f'Loading audio data at {args.pathDB}')
        print("Loading the training dataset")
        #print(f"############### NPHONES {nPhones}")
        trainDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqTrain,
                                    (phoneLabels, nPhones),
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader,
                                    MAX_SIZE_LOADED=args.max_size_loaded)
        print("Training dataset loaded")
        print("")

        print("Loading the validation dataset")
        valDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqVal,
                                    (phoneLabels, nPhones),
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader)
        print("Validation dataset loaded")
        print("")
    else:
        phoneLabels, nPhones = None, None
        trainDataset = None
        valDataset = None

    if seqCapture is not None:

        if args.path_phone_data:
            print("Loading the phone labels at " + args.path_phone_data)
            phoneLabelsForCapture, nPhonesCapture = parseSeqLabels(args.path_phone_data)
        else:
            assert not args.capturePhoneAlign
            phoneLabelsForCapture = None
            
        print("Loading the capture dataset")
        captureDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqCapture,
                                    (phoneLabelsForCapture, nPhonesCapture),
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader)
        print("Capture dataset loaded")
        print("")

        if args.captureSetStats:
            captureSetStatsCollector = statutil.constructStatCollectorFromSpecs(args.captureSetStats)
        else:
            captureSetStatsCollector = None
    else:
        captureDataset = None
        captureSetStatsCollector = None

    if args.FCMsettings:
        fcmSettings = {
            "FCMproject": args.FCMproject,
            "numProtos": args.FCMprotos, 
            "mBeforeAR": args.FCMmBeforeAR, 
            "leftProtos": args.FCMleaveProtos,
            "pushDegFeatureBeforeAR": args.FCMpushDegFeatureBeforeAR, 
            "mAfterAR": args.FCMmAfterAR,
            "pushDegCtxAfterAR": args.FCMpushDegCtxAfterAR,
            "pushDegAllAfterAR": args.FCMpushDegAllAfterAR,
            "reprsConcat": args.FCMreprsConcat, #,
            "reprsConcatNormSumsNotLengths": args.FCMreprsConcatNormSumsNotLengths,
            "pushLossWeightEnc": args.FCMpushLossWeightEnc,
            "pushLossWeightCtx": args.FCMpushLossWeightCtx,
            "VQpushEncCenterWeightOnTopConv": args.FCMVQpushEncCenterWeightOnTopConv,
            "VQpushEncCenterWeightOnlyAR": args.FCMVQpushEncCenterWeightOnlyAR,
            "VQpushEncCenterWeightOnlyCriterion": args.FCMVQpushEncCenterWeightOnlyCriterion,
            "VQgradualStart": args.FCMVQgradualStart,
            "VQpushCtxCenterWeight": args.FCMVQpushCtxCenterWeight,
            "pushLossLinear": args.FCMpushLossLinear,
            "pushLossGradualStart": args.FCMpushLossGradualStart,
            "pushLossProtosMult": args.FCMpushLossProtosMult,
            "pushLossCenterNorm": args.FCMpushLossCenterNorm,
            "pushLossPointNorm": args.FCMpushLossPointNorm,
            "pushLossNormReweight": args.FCMpushLossNormReweight,
            "hierARshorten": args.FCMhierARshorten,
            "hierARgradualStart": args.FCMhierARgradualStart,
            "hierARmergePrior": args.FCMhierARmergePrior,
            "modelLengthInAR": args.modelLengthInAR
            #"reprsConcatDontIncreaseARdim": args.FCMreprsConcatIncreaseARdim
        }
        # TODO: maybe better settings? or maybe ok
        if args.FCMcentermodule:
            centerInitSettings = {
                "mode": args.FCMcenter_mode,
                "numCentroids": args.FCMprotos,
                "reprDim": args.hiddenEncoder,
                "numPhones": nPhones,
                "initAfterEpoch": args.FCMcenter_initAfterEpoch,
                "firstInitNoIters": args.FCMcenter_firstInitNoIters,
                "kmeansInitIters": args.FCMcenter_kmeansInitIters,
                "kmeansInitBatches": args.FCMcenter_kmeansInitBatches,
                "kmeansReinitEachN": args.FCMcenter_kmeansReinitEachN,
                "kmeansReinitUpTo": args.FCMcenter_kmeansReinitUpTo,
                "onlineKmeansBatches": args.FCMcenter_onlineKmeansBatches,
                "onlineKmeansBatchesLongTerm": args.FCMcenter_onlineKmeansBatchesLongTerm,
                "onlineKmeansBatchesLongTermWeight": args.FCMcenter_onlineKmeansBatchesLongTermWeight,
                "centerNorm": args.FCMcenter_norm,
                "batchRecompute": args.FCMcenter_batchRecompute
            }
        else:
            centerInitSettings = None
        if args.FCMsegmentCostModule:
            segmentCostSettings = {
                "batchesMem": args.FCMsegment_batchesMem
            }
        else:
            segmentCostSettings = None
        if args.FCMleaveProtos is not None and args.FCMleaveProtos > 0:
            assert args.FCMleaveProtos <= args.FCMprotos
            args.FCMprotosForCriterion = args.FCMleaveProtos
        else:
            args.FCMprotosForCriterion = args.FCMprotos
    else:
        fcmSettings = None
        centerInitSettings = None
        segmentCostSettings = None

    #print(f'REPRCONCAT {fcmSettings["reprsConcat"]}')
    if fcmSettings is not None:  
        #locArgsCpy = deepcopy(locArgs)
        if fcmSettings["reprsConcat"]:
            assert fcmSettings["mBeforeAR"] is not None \
                or fcmSettings["mAfterAR"] is not None
            assert fcmSettings["pushDegFeatureBeforeAR"] is not None \
                or fcmSettings["pushDegCtxAfterAR"] is not None \
                or fcmSettings["pushDegAllAfterAR"] is not None
            # if fcmSettings["reprsConcatDontIncreaseARdim"] is not None:
            #     args.ARinputDim = args.hiddenEncoder + fcmSettings["numProtos"]
            #     args.hiddenGar = args.hiddenEncoder
            # else:
            if fcmSettings["mBeforeAR"] is not None:
                args.ARinputDim = args.hiddenEncoder + fcmSettings["numProtos"]
                args.hiddenGar = args.hiddenEncoder + fcmSettings["numProtos"]
            elif fcmSettings["mAfterAR"] is not None:
                args.ARinputDim = args.hiddenEncoder
                args.hiddenGar = args.hiddenEncoder
        elif fcmSettings["mBeforeAR"] is not None:
            args.ARinputDim = fcmSettings["numProtos"]
            args.hiddenGar = fcmSettings["numProtos"]
        elif fcmSettings["mAfterAR"] is not None:
            args.ARinputDim = args.hiddenEncoder
            pass  # in this case need to only pass changed dim to criterion,
                  # as there is no dim change inside encoder nor AR nets
        else:  # otherwise just pushing to closest proto, without dim change
            args.ARinputDim = args.hiddenEncoder
        print("FCM settings:", fcmSettings)
    else:
        #locArgsCpy = deepcopy(locArgs)
        args.ARinputDim = args.hiddenEncoder

    print(f"ARinputDim: {args.ARinputDim}")

    # here args are ready, can dump
    json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))

    segmentCostModel = None  # will be loaded or created if it is used
    if args.load is not None:
        if args.gru_level is not None and args.gru_level > 0:
            updateConfig = argparse.Namespace(nLevelsGRU=args.gru_level)
        else:
            updateConfig = None


        # loadBestNotLast = args.onlyCapture or args.only_classif_metric
        # could use this option for loading best state when not running actual training
        # but relying on CPC internal acc isn't very reliable
        # [!] caution - because of how they capture checkpoints,
        #     they capture "best in this part of training" as "best" (apart from capturing current state)
        #     so if best is in epoch 100 and training is paused and resumed from checkpoint
        #     in epoch 150, checkpoint from epoch 200 has "best from epoch 150" saved as globally best
        #     (but this is internal-CPC-score best anyway, which is quite vague)
        loadedData = \
            fl.loadModel(args.load, args.batchSizeGPU, fcmSettings=fcmSettings, load_nullspace=args.nullspace, updateConfig=updateConfig, loadSCM=(segmentCostSettings is not None))
        if segmentCostSettings is not None:
            cpcModel, args.hiddenGar, args.hiddenEncoder, segmentCostModel = loadedData
        else:
            cpcModel, args.hiddenGar, args.hiddenEncoder = loadedData
        CPChiddenGar, CPChiddenEncoder = args.hiddenGar, args.hiddenEncoder            

        if args.gru_level is not None and args.gru_level > 0:
            # Keep hidden units at LSTM layers on sequential batches
            if args.nullspace:
                cpcModel.cpc.gAR.keepHidden = True
            else:
                cpcModel.gAR.keepHidden = True

    else:
        # Encoder network
        encoderNet = fl.getEncoder(args)
        # AR Network
        arNet = fl.getAR(args)

        cpcModel = model.CPCModel(encoderNet, arNet, args.batchSizeGPU, fcmSettings=fcmSettings)

        CPChiddenGar, CPChiddenEncoder = cpcModel.gAR.getDimOutput(), cpcModel.gEncoder.getDimOutput()
    # TODO saving, loading, stuff for centerModel
    if centerInitSettings is not None:
        centerModel = center_model.CentroidModule(centerInitSettings)
    else:
        centerModel = None

    if segmentCostSettings is not None and segmentCostModel is None:
        segmentCostModel = SegmentCostModel(segmentCostSettings)
    # else already set to None, no need to change

    batchSize = args.nGPU * args.batchSizeGPU
    cpcModel.supervised = args.supervised

    downsampling = cpcModel.cpc.gEncoder.DOWNSAMPLING if isinstance(cpcModel, model.CPCModelNullspace) else cpcModel.gEncoder.DOWNSAMPLING
    # Training criterion
    if args.load is not None and args.loadCriterion:
        cpcCriterion = loadCriterion(args.load[0],  downsampling,
                                     len(speakers), nPhones)
    else:
        cpcCriterion = getCriterion(args, downsampling,
                                    len(speakers), nPhones)

    if loadOptimizer:
        state_dict = torch.load(args.load[0], 'cpu')
        cpcCriterion.load_state_dict(state_dict["cpcCriterion"])

    cpcCriterion.cuda()
    cpcModel.cuda()
    
    # Optimizer
    g_params = list(cpcCriterion.parameters()) + list(cpcModel.parameters())
    if centerModel is not None:
        g_params += list(centerModel.parameters())

    lr = args.learningRate
    optimizer = torch.optim.Adam(g_params, lr=lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    try:
        if loadOptimizer and not args.onlyCapture and not args.only_classif_metric:
            print("Loading optimizer " + args.load[0])
            state_dict = torch.load(args.load[0], 'cpu')
            #print("!!!", state_dict["optimizer"].keys(), state_dict["optimizer"]['state'].keys(), state_dict["optimizer"]['state'][183].keys(), len(state_dict["optimizer"]['state'].keys()), state_dict["optimizer"]['param_groups'])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
    except:
        print("--> WARNING: couldn't load optimizer state")  # can happen if adding / removing centerModel
        optimizer = torch.optim.Adam(g_params, lr=lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    # Checkpoint
    if args.pathCheckpoint is not None and not args.onlyCapture and not args.only_classif_metric:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")
        with open(args.pathCheckpoint + "_args.json", 'w') as file:
            json.dump(vars(args), file, indent=2)

    scheduler = None
    if args.schedulerStep > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.schedulerStep,
                                                    gamma=0.5)
    if args.schedulerRamp is not None:
        n_epoch = args.schedulerRamp
        print(f"Ramp activated. n_e = {n_epoch}")
        scheduler_ramp = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda epoch: utils.ramp_scheduling_function(
                                                               n_epoch, epoch),
                                                           last_epoch=-1)
        if scheduler is None:
            scheduler = scheduler_ramp
        else:
            scheduler = utils.SchedulerCombiner([scheduler_ramp, scheduler],
                                                [0, args.schedulerRamp])
    if scheduler is not None:
        print(f"DOING {len(range(cpcEpochCompleted + 1))} SCHEDULER STEPS")
        for i in range(cpcEpochCompleted + 1):  #len(logs["epoch"])):
            scheduler.step()

    print("cpcModel", cpcModel)
    print("cpcModelParams", list(map(lambda x: x.shape, cpcModel.parameters())))
    for name, param in cpcModel.named_parameters():
        if param.requires_grad:
            print(f"param: {name}, {param.data.shape}")
    print("cpcCriterion", cpcCriterion)

    cpcModel = torch.nn.DataParallel(cpcModel,
                                     device_ids=range(args.nGPU)).cuda()
    cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU)).cuda()
    
    if args.supervised_classif_metric:

        print("--> setting up linsep things")

        linsep_batch_size = args.linsepBatchSizeGPU * args.nGPU

        dim_features = CPChiddenEncoder if args.phone_get_encoded else CPChiddenGar
        dim_ctx_features = CPChiddenGar  # for speakers using CNN encodings is not supported; could add but not very useful perhaps
        if args.FCMmBeforeAR:
            dim_features = args.FCMprotos if args.phone_get_encoded else CPChiddenGar
            # ctx_features CPChiddenGar, as in this case it's just AR dim; it's also == args.FCMprotos
        elif args.FCMmAfterAR:
            # this case FCM is done at the end in CPC model on both features and ctx's
            dim_features = args.FCMprotos
            dim_ctx_features = args.FCMprotos

        #phoneLabelsData = None
        if args.path_phone_data:
            #phoneLabelsData, nPhonesInData = parseSeqLabels(args.path_phone_data)
            # ^ this is now done high above, and below same DS as above is used
            
            if not args.CTCphones:
                print(f"Running phone separability with aligned phones")
            else:
                print(f"Running phone separability with CTC loss")

            def constructPhoneCriterionAndOptimizer():
                if not args.CTCphones:
                    # print(f"Running phone separability with aligned phones")
                    phone_criterion = cr.PhoneCriterion(dim_features,
                                                nPhones, args.phone_get_encoded,
                                                nLayers=args.linsep_net_layers)
                else:
                    # print(f"Running phone separability with CTC loss")
                    phone_criterion = cr.CTCPhoneCriterion(dim_features,
                                                    nPhones, args.phone_get_encoded,
                                                    nLayers=args.linsep_net_layers)
                phone_criterion.cuda()
                phone_criterion = torch.nn.DataParallel(phone_criterion, device_ids=range(args.nGPU))

                # Optimizer
                phone_g_params = list(phone_criterion.parameters())

                phone_optimizer = torch.optim.Adam(phone_g_params, lr=args.linsep_lr,
                                            betas=(args.linsep_beta1, args.linsep_beta2),
                                            eps=args.linsep_epsilon)
                
                return phone_criterion, phone_optimizer
        
        if args.speaker_sep:
            print(f"Running speaker separability")

            def constructSpeakerCriterionAndOptimizer():
                speaker_criterion = cr.SpeakerCriterion(dim_ctx_features, len(speakers),
                                                        nLayers=args.linsep_net_layers)
                speaker_criterion.cuda()
                speaker_criterion = torch.nn.DataParallel(speaker_criterion, device_ids=range(args.nGPU))

                speaker_g_params = list(speaker_criterion.parameters())

                speaker_optimizer = torch.optim.Adam(speaker_g_params, lr=args.linsep_lr,
                                            betas=(args.linsep_beta1, args.linsep_beta2),
                                            eps=args.linsep_epsilon)

                return speaker_criterion, speaker_optimizer

        # print("preparing linsep DBs")

        # linsep_db_train = AudioBatchData(args.pathDB, args.sizeWindow, seqTrain,
        #                         phoneLabelsData, len(speakers),
        #                                 nProcessLoader=args.n_process_loader,
        #                                 MAX_SIZE_LOADED=args.max_size_loaded)

        # print("linsep_db_train ready")

        # linsep_db_val = AudioBatchData(args.pathDB, args.sizeWindow, seqVal,
        #                             phoneLabelsData, len(speakers),
        #                             nProcessLoader=args.n_process_loader,
        #                             MAX_SIZE_LOADED=args.max_size_loaded)

        # print("linsep_db_val ready")
        #linsep_db_train
        linsep_train_loader = trainDataset.getDataLoader(linsep_batch_size, "uniform", True,
                                        numWorkers=0)

        print("linsep_train_loader ready")
        # linsep_db_val
        linsep_val_loader = valDataset.getDataLoader(linsep_batch_size, 'sequential', False,
                                    numWorkers=0)

        print("linsep_val_loader ready")

        def runLinsepClassificationTraining(numOfEpoch, cpcMdl, centerModel, segmentCostModel, cpcStateEpochs):
            locLogsPhone = {}
            locLogsSpeaker = {}
            for linsepNr in range(args.linsep_times):
                log_path_for_epoch = os.path.join(args.linsep_logs_dir, str(numOfEpoch))
                if not os.path.exists(log_path_for_epoch):
                    os.makedirs(log_path_for_epoch)
                log_path_phoneme = os.path.join(log_path_for_epoch, "phoneme_linsep"+str(linsepNr)+"/")
                log_path_speaker = os.path.join(log_path_for_epoch, "speaker_linsep"+str(linsepNr)+"/")
                if not os.path.exists(log_path_phoneme):
                    os.makedirs(log_path_phoneme)
                if not os.path.exists(log_path_speaker):
                    os.makedirs(log_path_speaker)
                if args.linsep_checkpoint_dir:
                    checpoint_path_for_epoch = os.path.join(args.linsep_checkpoint_dir, str(numOfEpoch))
                    checkpoint_path_phoneme = os.path.join(checpoint_path_for_epoch, "phoneme_linsep"+str(linsepNr)+"/")
                    checkpoint_path_speaker = os.path.join(checpoint_path_for_epoch, "speaker_linsep"+str(linsepNr)+"/")
                    if not os.path.exists(checkpoint_path_phoneme):
                        os.makedirs(checkpoint_path_phoneme)
                    if not os.path.exists(checkpoint_path_speaker):
                        os.makedirs(checkpoint_path_speaker)
                locLogsPhone = {}
                locLogsSpeaker = {}
                if args.path_phone_data:
                    phone_criterion, phone_optimizer = constructPhoneCriterionAndOptimizer()
                    locLogsPhone = linsep.trainLinsepClassification(
                        cpcMdl,
                        centerModel,
                        segmentCostModel,
                        phone_criterion,  # combined with classification model before
                        linsep_train_loader,
                        linsep_val_loader,
                        phone_optimizer,
                        log_path_phoneme,
                        args.linsep_task_logging_step,
                        checkpoint_path_phoneme,
                        args.linsep_n_epoch,
                        cpcStateEpochs,
                        'phone')
                    del phone_criterion
                    del phone_optimizer
                if args.speaker_sep:
                    speaker_criterion, speaker_optimizer = constructSpeakerCriterionAndOptimizer()
                    locLogsSpeaker = linsep.trainLinsepClassification(
                        cpcMdl,
                        centerModel,
                        segmentCostModel,
                        speaker_criterion,  # combined with classification model before
                        linsep_train_loader,
                        linsep_val_loader,
                        speaker_optimizer,
                        log_path_speaker,
                        args.linsep_task_logging_step,
                        checkpoint_path_speaker,
                        args.linsep_n_epoch,
                        cpcStateEpochs,
                        'speaker')
                    del speaker_criterion
                    del speaker_optimizer

                locLogsPhone = {**locLogsPhone, **{"linsep"+str(linsepNr)+"_phone_" + k: v for k, v in locLogsPhone.items()}}
                locLogsSpeaker = {**locLogsSpeaker, **{"linsep"+str(linsepNr)+"_speaker_" + k: v for k, v in locLogsSpeaker.items()}}
            return {**locLogsPhone, **locLogsSpeaker}

        linsepClassificationTaskConfig = (args.linsep_classif_each_epochs,
                                            runLinsepClassificationTraining)

    else:
        linsepClassificationTaskConfig = (None, None)

    print("-------> starting actual run <-------")

    if not args.onlyCapture and not args.only_classif_metric:
        run(trainDataset,
            valDataset,
            (captureDataset, captureOptions, captureSetStatsCollector),
            linsepClassificationTaskConfig,
            batchSize,
            args.samplingType,
            cpcModel,
            centerModel,
            segmentCostModel,
            cpcCriterion,
            args.nEpoch,
            args.pathCheckpoint,
            optimizer,
            scheduler,
            logs,
            cpcEpochCompleted)
    if args.onlyCapture:  
    # caution [!] - will capture for last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        onlyCapture(
            (captureDataset, captureOptions, captureSetStatsCollector),
            batchSize,
            cpcModel,
            centerModel,
            segmentCostModel,
            cpcCriterion,
            logs,
            cpcEpochCompleted)
    if args.only_classif_metric:
    # caution [!] - will use last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        trainedEpoch = cpcEpochCompleted  #len(logs["epoch"]) - 1
        # runPhonemeClassificationTraining created above if args.supervised_classif_metric
        runLinsepClassificationTraining(trainedEpoch, cpcModel, centerModel, segmentCostModel, (trainedEpoch, args.nEpoch))


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Trainer')

    print(len(argv))

    # Default arguments:
    parser = set_default_cpc_config(parser)

    group_db = parser.add_argument_group('Dataset')
    group_db.add_argument('--pathDB', type=str, default=None,
                          help='Path to the directory containing the '
                          'data.')
    group_db.add_argument('--file_extension', type=str, default=".flac",
                          help="Extension of the audio files in the dataset.")
    group_db.add_argument('--pathTrain', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'training sequences.')
    group_db.add_argument('--pathVal', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'validation sequences.')
    # stuff below for capturing data
    group_db.add_argument('--onlyCapture', action='store_true',
                          help='Only capture data from learned model for one epoch, ignore training; '
                          'conflicts with pathTrain, pathVal etc. arguments')
    group_db.add_argument('--pathCaptureDS', type=str, default=None,
                          help='Path to a .txt file containing the list of the '
                          'data capturing sequences; additionally it can be specified to log a total number of N, or n percent of set '
                          '(e.g. pass validation path and specify to sample from that)')
    group_db.add_argument('--captureDSfreq', type=int, default=None,
                          help='percentage of pathCaptureDS set to use for capturing; conflicts with --captureDStotNr')
    group_db.add_argument('--captureDStotNr', type=int, default=None,
                          help='total number of *AUDIO FILES* to capture data for; number of chunks will be different.')
    # end of capturing data part here
    group_db.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    group_db.add_argument('--ignore_cache', action='store_true',
                          help='Activate if the dataset has been modified '
                          'since the last training session.')
    group_db.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    group_db.add_argument('--gru_level', type=int, default=-1,
                          help='Hidden level of the LSTM autoregressive model to be taken'
                          '(default: -1, last layer).')

    group_supervised = parser.add_argument_group(
        'Supervised mode (depreciated)')
    group_supervised.add_argument('--supervised', action='store_true',
                                  help='(Depreciated) Disable the CPC loss and activate '
                                  'the supervised mode. By default, the supervised '
                                  'training method is the speaker classification.')
    # group_supervised.add_argument('--pathPhone', type=str, default=None,
    #                               help='(Supervised mode only) Path to a .txt '
    #                               'containing the phone labels of the dataset. If given '
    #                               'and --supervised, will train the model using a '
    #                               'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_supervised_data = parser.add_argument_group(
        'Group with args for passing supervised data both for additional metric-producing classification task, '
        'and for data capturing')
    group_supervised_data.add_argument('--path_phone_data', type=str, default=None,
                        help="Path to the phone labels. If given, with --supervised_classif_metric will be able "
                        'to learn phone classification, with capturing will be able to capture phone alignments')

    group_supervised_metric = parser.add_argument_group(
        'Mode with computing additional supervised phoneme classification accuracy, withou influencing CPC training')
    group_supervised_metric.add_argument('--supervised_classif_metric',
                        action='store_true', help='Compute the metric')
    group_supervised_metric.add_argument('--linsep_times',
                        type=int, default=1, help='number of linseps (for result avg stability)')
    group_supervised_metric.add_argument('--speaker_sep', action='store_true',
                        help="If given, will"
                        " compute the speaker separability.")
    group_supervised_metric.add_argument('--CTCphones', action='store_true',
                        help="Use the CTC loss (for phone separability only)")
    group_supervised_metric.add_argument('--linsepBatchSizeGPU', type=int, default=8,
                        help='Batch size per GPU for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_n_epoch', type=int, default=10)
    group_supervised_metric.add_argument('--phone_get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    group_supervised_metric.add_argument('--linsep_lr', type=float, default=2e-4,
                        help='Learning rate for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_beta1', type=float, default=0.9,
                        help='Value of beta1 for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_beta2', type=float, default=0.999,
                        help='Value of beta2 for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--linsep_epsilon', type=float, default=2e-8,
                        help='Value of epsilon for the Adam optimizer for phoneme classification.')
    group_supervised_metric.add_argument('--only_classif_metric',
                        action="store_true", 
                        help="Don't train CPC, just compute classification accuracy on given checkpoint "
                        '(classification net itself is trained) and store in given path; '
                        'conflicts with regular CPC training; need to specify --supervised_classif_metric '
                        'and corresponding args')
    group_supervised_metric.add_argument('--linsep_logs_dir', type=str, default=None,
                        help='Path (root) where to log more detailed phoneme classification training data.')
    group_supervised_metric.add_argument('--linsep_checkpoint_dir', type=str, default=None,
                        help='Path (root) where to save best checkpoint for each classification training performed.')
    group_supervised_metric.add_argument('--linsep_task_logging_step', type=int, default=1,
                        help='how often to save detailed phoneme classification training data')
    group_supervised_metric.add_argument('--linsep_classif_each_epochs', type=str, default="20",
                        help='How often to perform classification task - classification net is then '
                        'trained on train DS representations and assesed on val DS representations '
                        'that are produced after that epoch in eval mode')
    group_supervised_metric.add_argument('--linsep_net_layers', type=int, default='1',
                        help='Description of how big net to use for classification (layers have num_phonemes neurons) ' 
                        'with 1, there is just a linear net used without additional hidden layers')
    
    group_stats = parser.add_argument_group(
        'Args for specifying stats to compute for validation and capture DS')
    # group_stats.add_argument('--valSetStats', type=str, default=None,
    #                     help='For validation DS.')
    # validation DS has smaller number of info - will need to specify stats accordingly
    group_stats.add_argument('--captureSetStats', type=str, default=None,
                        help='For capture DS.')

    group_save = parser.add_argument_group('Save')
    group_save.add_argument('--pathCheckpoint', type=str, default=None,
                            help="Path of the output directory.")
    group_save.add_argument('--logging_step', type=int, default=1000)
    group_save.add_argument('--save_step', type=int, default=5,
                            help="Frequency (in epochs) at which a checkpoint "
                            "should be saved")

    # stuff below for capturing data
    group_save.add_argument('--pathCaptureSave', type=str, default=None, )
    group_save.add_argument('--captureEachEpochs', type=int, default=10, help='how often to save capture data')
    group_save.add_argument('--captureConvRepr', action='store_true', help='if to save representations after the encoder')
    group_save.add_argument('--captureCtxRepr', action='store_true', help='if to save LSTM-based contexts produced in CPC model')
    group_save.add_argument('--captureSpeakerAlign', action='store_true', help='if to save speaker alignments')
    group_save.add_argument('--capturePhoneAlign', action='store_true', help='if to save phone alignments')
    group_save.add_argument('--captureEverything', action='store_true', help='save everything valid in this config')
    # below ONLY for CPC-CTC
    group_save.add_argument('--capturePred', action='store_true', help='if to save CPC predictions')
    group_save.add_argument('--captureCPCCTCalign', action='store_true', help='if to save CTC alignments with CPC predictions - only for CPC-CTC variant')
    group_save.add_argument('--captureCPCCTClogScores', action='store_true', help='if to save alignment log scores')
    # end of capturing data part here

    group_load = parser.add_argument_group('Load')
    group_load.add_argument('--load', type=str, default=None, nargs='*',
                            help="Load an exsiting checkpoint. Should give a path "
                            "to a .pt file. The directory containing the file to "
                            "load should also have a 'checkpoint.logs' and a "
                            "'checkpoint.args'")
    group_load.add_argument('--loadCriterion', action='store_true',
                            help="If --load is activated, load the state of the "
                            "training criterion as well as the state of the "
                            "feature network (encoder + AR)")
    group_load.add_argument('--restart', action='store_true',
                            help="If any checkpoint is found, ignore it and "
                            "restart the training from scratch.")
    group_load.add_argument('--overrideArgsFile', action='store_true', help="override args from config file with passed values")
    group_load.add_argument('--nullspace', action='store_true',
                            help="Additionally load nullspace")

    group_fcm = parser.add_argument_group("FCM")
    group_fcm.add_argument('--FCMproject', action='store_true')
    group_fcm.add_argument('--FCMsettings', action='store_true')
    group_fcm.add_argument('--FCMprotos', type=int, default=50)
    group_fcm.add_argument('--FCMmBeforeAR', type=float, default=None)
    group_fcm.add_argument('--FCMleaveProtos', type=int, default=None)  # only makes sense for mBeforeAR
    group_fcm.add_argument('--FCMpushDegFeatureBeforeAR', type=float, default=None)
    group_fcm.add_argument('--FCMmAfterAR', type=float, default=None)
    group_fcm.add_argument('--FCMpushDegCtxAfterAR', type=float, default=None)
    group_fcm.add_argument('--FCMpushDegAllAfterAR', type=float, default=None)
    group_fcm.add_argument('--FCMreprsConcat', action='store_true')
    group_fcm.add_argument('--FCMreprsConcatNormSumsNotLengths', action='store_true')
    # TODO? one below for now is not properly done when AR=transformer
    #      WHEN AR=TRANSFORMER IT'S LIKE IT'S ALWAYS FALSE
    # TODO? this is also not properly dealt with for criterion's prediction network
    # [!!] --> actually, rather just keep it like that, and make smaller enc if needed
    #group_fcm.add_argument('--FCMreprsConcatDontIncreaseARdim', action='store_true')
    group_fcm.add_argument('--FCMpushLossWeightEnc', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMpushLossWeightCtx', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMVQpushEncCenterWeightOnTopConv', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMVQpushEncCenterWeightOnlyAR', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMVQpushEncCenterWeightOnlyCriterion', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMVQpushCtxCenterWeight', type=float, default=None)  # not really FCM part but well
    group_fcm.add_argument('--FCMVQgradualStart', type=int, default=None)  # not really FCM part but well
    # TODO think about adding linear loss option, but don't think makes too much sense?
    group_fcm.add_argument('--FCMpushLossLinear', action='store_true')
    group_fcm.add_argument('--FCMpushLossGradualStart', type=int, default=None)  # increase loss weight from 0 * x at chosen start epoch to 1 * x through the training
    group_fcm.add_argument('--FCMpushLossProtosMult', type=float, default=None)  # like VQ-VAE commitment loss
    group_fcm.add_argument('--FCMpushLossCenterNorm', action='store_true')
    group_fcm.add_argument('--FCMpushLossPointNorm', action='store_true')
    group_fcm.add_argument('--FCMpushLossNormReweight', action='store_true')
    group_fcm.add_argument('--FCMhierARshorten', type=float, default=None)  # how big length reduction to make
    group_fcm.add_argument('--FCMhierARgradualStart', type=int, default=None)
    group_fcm.add_argument('--FCMhierARmergePrior', type=str, default="se")  # how big length reduction to make

    group_fcm.add_argument('--FCMcentermodule', action='store_true')
    group_fcm.add_argument('--FCMcenter_mode', type=str, default=None)
    group_fcm.add_argument('--FCMcenter_initAfterEpoch', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_firstInitNoIters', action='store_true')
    group_fcm.add_argument('--FCMcenter_kmeansInitIters', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_kmeansInitBatches', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_kmeansReinitEachN', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_kmeansReinitUpTo', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_onlineKmeansBatches', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_onlineKmeansBatchesLongTerm', type=int, default=None)
    group_fcm.add_argument('--FCMcenter_onlineKmeansBatchesLongTermWeight', type=float, default=None)
    group_fcm.add_argument('--FCMcenter_norm', action='store_true')
    group_fcm.add_argument('--FCMcenter_batchRecompute', type=int, default=None)
    #FCMcenterInitAfterEpoch

    group_fcm.add_argument('--FCMsegmentCostModule', action='store_true')
    group_fcm.add_argument('--FCMsegment_batchesMem', type=int, default=None)

    group_fcm.add_argument('--modelLengthInAR', action='store_true')
    

    group_gpu = parser.add_argument_group('GPUs')
    group_gpu.add_argument('--nGPU', type=int, default=-1,
                           help="Number of GPU to use (default: use all "
                           "available GPUs)")
    group_gpu.add_argument('--batchSizeGPU', type=int, default=8,
                           help='Number of batches per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    args = parser.parse_args(argv)

    

    if args.pathDB is None and (args.pathCheckpoint is None or args.restart):
        parser.print_help()
        print("Either provides an input dataset or a checkpoint to load")
        sys.exit()

    if args.pathCheckpoint is not None:
        args.pathCheckpoint = os.path.abspath(args.pathCheckpoint)

    if args.load is not None:
        args.load = [os.path.abspath(x) for x in args.load]

    # set it up if needed, so that it is dumped along with other args
    if args.random_seed is None:
        args.random_seed = random.randint(0, 2**31)

    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    assert args.nGPU <= torch.cuda.device_count(),\
        f"number of GPU asked: {args.nGPU}," \
        f"number GPU detected: {torch.cuda.device_count()}"
    print(f"Let's use {args.nGPU} GPUs!")

    if args.arMode == 'no_ar':
        args.hiddenGar = args.hiddenEncoder
    return args


if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()

    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
