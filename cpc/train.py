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

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.model as model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
import cpc.eval.linear_separability as linsep
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
import cpc.stats.stat_utils as statutil


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
        else:
            sizeInputSeq = (args.sizeWindow // downsampling)

            if args.FCMproject and args.FCMmBeforeAR:
                # this could be replaced with encoder.getOutDim, but didn't want to change signature
                encoderOutDimForCriterion = args.FCMprotos
                # [!] args.hiddenGar already updated with possible FCM dim stuff in main train script
                #     it is the actual AR dimension - so only needs to be changed here in case of FCMmAfterAR
                #     changes the dimension after AR
                ARoutDimForCriterion = args.hiddenGar  # but actually should also be == args.FCMprotos

            elif args.FCMproject and args.FCMmAfterAR:
                encoderOutDimForCriterion = args.FCMprotos
                ARoutDimForCriterion = args.FCMprotos

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
                                                        sizeInputSeq=sizeInputSeq)
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
    _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
    criterion = getCriterion(locArgs, downsampling, nSpeakers, nPhones)

    state_dict = torch.load(pathCheckpoint, 'cpu')

    criterion.load_state_dict(state_dict["cpcCriterion"])
    return criterion


def trainStep(dataLoader,
              cpcModel,
              cpcCriterion,
              optimizer,
              scheduler,
              loggingStep):

    cpcModel.train()
    cpcCriterion.train()

    start_time = time.perf_counter()
    n_examples = 0
    logs, lastlogs = {}, None
    iter = 0
    for step, fulldata in enumerate(dataLoader):
        batchData, labelData = fulldata
        label = labelData['speaker']
        n_examples += batchData.size(0)
        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        c_feature, encoded_data, label = cpcModel(batchData, label)
        allLosses, allAcc, _ = cpcCriterion(c_feature, encoded_data, label, None)
        totLoss = allLosses.sum()

        totLoss.backward()

        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        if "locLoss_train" not in logs:
            logs["locLoss_train"] = np.zeros(allLosses.size(1))
            logs["locAcc_train"] = np.zeros(allLosses.size(1))

        iter += 1
        logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()

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
            start_time, n_examples = new_time, 0

    if scheduler is not None:
        scheduler.step()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Average training loss on epoch", logs)
    return logs


def valStep(dataLoader,
            cpcModel,
            cpcCriterion):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    for step, fulldata in enumerate(dataLoader):

        batchData, labelData = fulldata
        label = labelData['speaker']

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with torch.no_grad():
            c_feature, encoded_data, label = cpcModel(batchData, label)
            allLosses, allAcc, _ = cpcCriterion(c_feature, encoded_data, label, None)

        if "locLoss_val" not in logs:
            logs["locLoss_val"] = np.zeros(allLosses.size(1))
            logs["locAcc_val"] = np.zeros(allLosses.size(1))

        iter += 1
        logs["locLoss_val"] += allLosses.mean(dim=0).cpu().numpy()
        logs["locAcc_val"] += allAcc.mean(dim=0).cpu().numpy()

    logs = utils.update_logs(logs, iter)
    logs["iter"] = iter
    utils.show_logs("Validation loss:", logs)
    return logs


def captureStep(
            dataLoader,
            cpcModel,
            cpcCriterion,
            captureOptions,
            captureStatsCollector,
            epochNr):

    cpcCriterion.eval()
    cpcModel.eval()
    logs = {}
    cpcCriterion.eval()
    cpcModel.eval()
    iter = 0

    capturePath = captureOptions['path']
    whatToSave = captureOptions['what']
    cpcCaptureOpts = []
    if 'pred' in whatToSave:
        cpcCaptureOpts.append('pred')
    if 'cpcctc_align' in whatToSave:
        cpcCaptureOpts.append('cpcctc_align')

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

            c_feature, encoded_data, labelSpeaker = cpcModel(batchData, labelSpeaker)
            allLosses, allAcc, captured = cpcCriterion(c_feature, encoded_data, labelSpeaker, cpcCaptureOpts)
        
            # saving it with IDs like that assumes deterministic order of elements
            # which is there as dataLoader is a sequential one here
            if 'repr' in whatToSave:
                # encoded data shape: batch_size x len x repr_dim
                torch.save(encoded_data.cpu(), os.path.join(epochDir, 'repr', f'repr_batch{batchBegin}-{batchEnd}.pt'))
            if 'ctx' in whatToSave:
                # ctx data shape: also batch_size x len x repr_dim
                torch.save(c_feature.cpu(), os.path.join(epochDir, 'ctx', f'ctx_batch{batchBegin}-{batchEnd}.pt'))
            if 'speaker_align' in whatToSave:
                # speaker data shape: batch_size (1-dim, each one in batch is whole by 1 speaker)
                torch.save(labelSpeaker.cpu(), os.path.join(epochDir, 'speaker_align', f'speaker_align_batch{batchBegin}-{batchEnd}.pt'))
            if 'phone_align' in whatToSave:
                # phone alignment data shape: batch_size x len
                torch.save(labelData['phone'].cpu(), os.path.join(epochDir, 'phone_align', f'phone_align_batch{batchBegin}-{batchEnd}.pt'))
            for cpcCaptureThing in cpcCaptureOpts:
                # pred shape (CPC-CTC): batch_size x (len - num_matched) x repr_dim x num_predicts (or num_predicts +1 if self loop allowed)
                # align shape (CPC-CTC): batch_size x (len - num_matched) x num_matched
                torch.save(captured[cpcCaptureThing].cpu(), os.path.join(epochDir, cpcCaptureThing, 
                            f'{cpcCaptureThing}_batch{batchBegin}-{batchEnd}.pt'))

            if captureStatsCollector:
                allBatchData = {}
                allBatchData['repr'] = encoded_data
                allBatchData['ctx'] = c_feature
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
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler,
        logs):

    startEpoch = len(logs["epoch"])
    print(f"Running {nEpoch} epochs, now at {startEpoch}")
    bestAcc = 0
    bestStateDict = None
    start_time = time.time()
    
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    linsepEachEpochs, linsepFun = linsepClassificationTaskConfig
    assert (captureDataset is None and captureOptions is None) \
        or (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']

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

        locLogsTrain = trainStep(trainLoader, cpcModel, cpcCriterion,
                                optimizer, scheduler, logs["logging_step"])

        locLogsVal = valStep(valLoader, cpcModel, cpcCriterion)

        if captureDataset is not None and epoch % captureEachEpochs == 0:
            print(f"Capturing data for epoch {epoch}")
            captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, captureStatsCollector, epoch)

        currentAccuracy = float(locLogsVal["locAcc_val"].mean())
        if currentAccuracy > bestAcc:
            bestStateDict = deepcopy(fl.get_module(cpcModel).state_dict())  

        locLogsLinsep = {}
        # this performs linsep task for the best CPC model up to date
        if linsepEachEpochs is not None and epoch !=0 and epoch % linsepEachEpochs == 0:
            # capturing for current CPC state after this epoch, relying on CPC internal accuracy is vague
            locLogsLinsep = linsepFun(epoch, cpcModel, epoch)

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
                            f"{pathCheckpoint}_{epoch}.pt")
            utils.save_logs(logs, pathCheckpoint + "_logs.json")


def onlyCapture(
        captureDatasetWithOptions,
        batchSize,
        cpcModel,
        cpcCriterion,
        logs
):
    startEpoch = len(logs["epoch"])
    captureDataset, captureOptions, captureStatsCollector = captureDatasetWithOptions
    assert (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']
    print(f'Capture DS size: {str(len(captureDataset))}')

    captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                numWorkers=0)
    print(f"Capturing data for model checkpoint after epoch: {startEpoch-1}")
    captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, captureStatsCollector, startEpoch-1)


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
    # needed to move thing below later, as adding some args later
    #json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))
    if args.pathCheckpoint is not None and not args.restart:
        cdata = fl.getCheckpointData(args.pathCheckpoint)
        if cdata is not None:
            data, logs, locArgs = cdata
            print(f"Checkpoint detected at {data}")
            fl.loadArgs(args, locArgs,
                        forbiddenAttr={"nGPU", "pathCheckpoint",
                                       "debug", "restart", "world_size",
                                       "n_nodes", "node_id", "n_gpu_per_node",
                                       "max_size_loaded"})
            args.load, loadOptimizer = [data], True
            args.loadCriterion = True

    logs["logging_step"] = args.logging_step

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
            whatToSave = ['repr', 'ctx', 'speaker_align', 'pred']
            if args.path_phone_data:
                whatToSave.append('phone_align')
            if args.CPCCTC:
                whatToSave.append('cpcctc_align')
        else:
            for argVal, name in zip([args.captureRepr, 
                                    args.captureCtx, 
                                    args.captureSpeakerAlign, 
                                    args.capturePhoneAlign,
                                    args.capturePred,
                                    args.captureCPCCTCalign], 
                                    ['repr', 'ctx', 'speaker_align', 'phone_align', 'pred', 'cpcctc_align']):
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
        trainDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqTrain,
                                    phoneLabels,
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader,
                                    MAX_SIZE_LOADED=args.max_size_loaded)
        print("Training dataset loaded")
        print("")

        print("Loading the validation dataset")
        valDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqVal,
                                    phoneLabels,
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
            phoneLabelsForCapture, _ = parseSeqLabels(args.path_phone_data)
        else:
            assert not args.capturePhoneAlign
            phoneLabelsForCapture = None
            
        print("Loading the capture dataset")
        captureDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqCapture,
                                    phoneLabelsForCapture,
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

    if args.FCMproject:
        fcmSettings = {
            "numProtos": args.FCMprotos, 
            "mBeforeAR": args.FCMmBeforeAR, 
            "leftProtos": args.FCMleaveProtos,
            "pushDegFeatureBeforeAR": args.FCMpushDegFeatureBeforeAR, 
            "mAfterAR": args.FCMmAfterAR,
            "pushDegCtxAfterAR": args.FCMpushDegCtxAfterAR,
            "pushDegAllAfterAR": args.FCMpushDegAllAfterAR
        }
    else:
        fcmSettings = None

    if fcmSettings is not None:  
        #locArgsCpy = deepcopy(locArgs)
        if fcmSettings["mBeforeAR"] is not None:
            args.ARinputDim = fcmSettings["numProtos"]
            args.hiddenGar = fcmSettings["numProtos"]
        elif fcmSettings["mAfterAR"] is not None:
            args.ARinputDim = args.hiddenEncoder
            pass  # in this case need to only pass changed dim to criterion,
                  # as there is no dim change inside encoder nor AR nets
        else:  # otherwise just pushing to closest proto, without dim change
            args.ARinputDim = args.hiddenEncoder
    else:
        #locArgsCpy = deepcopy(locArgs)
        args.ARinputDim = args.hiddenEncoder

    print(f"ARinputDim: {args.ARinputDim}")

    # here args are ready, can dump
    json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))

    if args.load is not None:
        # loadBestNotLast = args.onlyCapture or args.only_classif_metric
        # could use this option for loading best state when not running actual training
        # but relying on CPC internal acc isn't very reliable
        # [!] caution - because of how they capture checkpoints,
        #     they capture "best in this part of training" as "best" (apart from capturing current state)
        #     so if best is in epoch 100 and training is paused and resumed from checkpoint
        #     in epoch 150, checkpoint from epoch 200 has "best from epoch 150" saved as globally best
        #     (but this is internal-CPC-score best anyway, which is quite vague)
        cpcModel, args.hiddenGar, args.hiddenEncoder = \
            fl.loadModel(args.load, fcmSettings=fcmSettings)
        CPChiddenGar, CPChiddenEncoder = args.hiddenGar, args.hiddenEncoder
    else:
        # Encoder network
        encoderNet = fl.getEncoder(args)
        # AR Network
        arNet = fl.getAR(args)

        cpcModel = model.CPCModel(encoderNet, arNet, fcmSettings=fcmSettings)

        CPChiddenGar, CPChiddenEncoder = cpcModel.gAR.getDimOutput(), cpcModel.gEncoder.getDimOutput()

    batchSize = args.nGPU * args.batchSizeGPU
    cpcModel.supervised = args.supervised

    # Training criterion
    if args.load is not None and args.loadCriterion:
        cpcCriterion = loadCriterion(args.load[0], cpcModel.gEncoder.DOWNSAMPLING,
                                     len(speakers), nPhones)
    else:
        cpcCriterion = getCriterion(args, cpcModel.gEncoder.DOWNSAMPLING,
                                    len(speakers), nPhones)

    if loadOptimizer:
        state_dict = torch.load(args.load[0], 'cpu')
        cpcCriterion.load_state_dict(state_dict["cpcCriterion"])

    cpcCriterion.cuda()
    cpcModel.cuda()
    
    # Optimizer
    g_params = list(cpcCriterion.parameters()) + list(cpcModel.parameters())

    lr = args.learningRate
    optimizer = torch.optim.Adam(g_params, lr=lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    if loadOptimizer and not args.onlyCapture and not args.only_classif_metric:
        print("Loading optimizer " + args.load[0])
        state_dict = torch.load(args.load[0], 'cpu')
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])

    # Checkpoint
    if args.pathCheckpoint is not None:
        if not os.path.isdir(args.pathCheckpoint):
            os.mkdir(args.pathCheckpoint)
        args.pathCheckpoint = os.path.join(args.pathCheckpoint, "checkpoint")

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
        for i in range(len(logs["epoch"])):
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

        def runLinsepClassificationTraining(numOfEpoch, cpcMdl, cpcStateEpoch):
            log_path_for_epoch = os.path.join(args.linsep_logs_dir, str(numOfEpoch))
            if not os.path.exists(log_path_for_epoch):
                os.makedirs(log_path_for_epoch)
            log_path_phoneme = os.path.join(log_path_for_epoch, "phoneme/")
            log_path_speaker = os.path.join(log_path_for_epoch, "speaker/")
            if not os.path.exists(log_path_phoneme):
                os.makedirs(log_path_phoneme)
            if not os.path.exists(log_path_speaker):
                os.makedirs(log_path_speaker)
            if args.linsep_checkpoint_dir:
                checpoint_path_for_epoch = os.path.join(args.linsep_checkpoint_dir, str(numOfEpoch))
                checkpoint_path_phoneme = os.path.join(checpoint_path_for_epoch, "phoneme/")
                checkpoint_path_speaker = os.path.join(checpoint_path_for_epoch, "speaker/")
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
                    phone_criterion,  # combined with classification model before
                    linsep_train_loader,
                    linsep_val_loader,
                    phone_optimizer,
                    log_path_phoneme,
                    args.linsep_task_logging_step,
                    checkpoint_path_phoneme,
                    args.linsep_n_epoch,
                    cpcStateEpoch,
                    'phone')
                del phone_criterion
                del phone_optimizer
            if args.speaker_sep:
                speaker_criterion, speaker_optimizer = constructSpeakerCriterionAndOptimizer()
                locLogsSpeaker = linsep.trainLinsepClassification(
                    cpcMdl,
                    speaker_criterion,  # combined with classification model before
                    linsep_train_loader,
                    linsep_val_loader,
                    speaker_optimizer,
                    log_path_speaker,
                    args.linsep_task_logging_step,
                    checkpoint_path_speaker,
                    args.linsep_n_epoch,
                    cpcStateEpoch,
                    'speaker')
                del speaker_criterion
                del speaker_optimizer

            locLogsPhone = {"phone_" + k: v for k, v in locLogsPhone.items()}
            locLogsSpeaker = {"speaker_" + k: v for k, v in locLogsSpeaker.items()}
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
            cpcCriterion,
            args.nEpoch,
            args.pathCheckpoint,
            optimizer,
            scheduler,
            logs)
    if args.onlyCapture:  
    # caution [!] - will capture for last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        onlyCapture(
            (captureDataset, captureOptions, captureSetStatsCollector),
            batchSize,
            cpcModel,
            cpcCriterion,
            logs)
    if args.only_classif_metric:
    # caution [!] - will use last checkpoint (last saved state) if checkpoint directory given
    #               to use specific checkpoint provide full checkpoint file path
    #               will use "last state" and not "best in internal CPC accuracy" anyway
        trainedEpoch = len(logs["epoch"]) - 1
        # runPhonemeClassificationTraining created above if args.supervised_classif_metric
        runLinsepClassificationTraining(trainedEpoch, cpcModel, trainedEpoch)


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
    group_supervised_metric.add_argument('--linsep_classif_each_epochs', type=int, default=20,
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
    group_save.add_argument('--captureRepr', action='store_true', help='if to save representations after the encoder')
    group_save.add_argument('--captureCtx', action='store_true', help='if to save LSTM-based contexts produced in CPC model')
    group_save.add_argument('--captureSpeakerAlign', action='store_true', help='if to save speaker alignments')
    group_save.add_argument('--capturePhoneAlign', action='store_true', help='if to save phone alignments')
    # below ONLY for CPC-CTC
    group_save.add_argument('--capturePred', action='store_true', help='if to save CPC predictions')
    group_save.add_argument('--captureCPCCTCalign', action='store_true', help='if to save CTC alignments with CPC predictions - only for CPC-CTC variant')
    group_save.add_argument('--captureEverything', action='store_true', help='save everythong valid in this config')
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

    group_fcm = parser.add_argument_group("FCM")
    group_fcm.add_argument('--FCMproject', action='store_true')
    group_fcm.add_argument('--FCMprotos', type=int, default=50)
    group_fcm.add_argument('--FCMmBeforeAR', type=float, default=None)
    group_fcm.add_argument('--FCMleaveProtos', type=int, default=None)  # only makes sense for mBeforeAR
    group_fcm.add_argument('--FCMpushDegFeatureBeforeAR', type=float, default=None)
    group_fcm.add_argument('--FCMmAfterAR', type=float, default=None)
    group_fcm.add_argument('--FCMpushDegCtxAfterAR', type=float, default=None)
    group_fcm.add_argument('--FCMpushDegAllAfterAR', type=float, default=None)
    

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
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
