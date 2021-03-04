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
#import psutil
import sys
import math

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.model as model
import cpc.utils.misc as utils
import cpc.feature_loader as fl
from cpc.cpc_default_config import set_default_cpc_config
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels


def getCriterion(args, downsampling, nSpeakers, nPhones):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if not args.supervised:
        if args.cpc_mode == 'none':
            cpcCriterion = cr.NoneCriterion()
        else:
            sizeInputSeq = (args.sizeWindow // downsampling)
            if args.CPCCTC:
                cpcCriterion = sa.CPCUnsupersivedCriterion(args.nPredicts,
                                                        args.CPCCTCNumMatched,
                                                        args.hiddenGar,
                                                        args.hiddenEncoder,
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
                                                        args.hiddenGar,
                                                        args.hiddenEncoder,
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
        batchData, label = fulldata
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

def removeInvalidClassifData(sample, labels):
    validIndices = []
    for i in range(labels.shape[0]):
        isInvalid = torch.min(labels[i]) < 0
        if not isInvalid:
            validIndices.append(i)
    if len(validIndices) == labels.shape[0]:
        return sample, labels, labels.shape[0], 0
    fixedSample = torch.zeros((len(validIndices), *(sample.shape[1:])), dtype=sample.dtype)
    fixedLabels = torch.zeros((len(validIndices), *(labels.shape[1:])), dtype=labels.dtype)
    for i, idx in enumerate(validIndices):
        fixedSample[i] = sample[idx]
        fixedLabels[i] = labels[idx]
    return fixedSample, fixedLabels, len(validIndices), labels.shape[0] - len(validIndices)  # last thing are invalid lines


def trainClassifStep(dataLoader,
              encoderNet,
              classifModel,
              classifCriterion,
              optimizer): #,
              #scheduler,
              #loggingStep):

    #cpcModel.eval()
    encoderNet.eval()
    classifModel.train()
    classifCriterion.train()

    #start_time = time.perf_counter()   # optimizer.state_dict()['state'][0]['exp_avg'].min()
    n_examples = 0
    bad_lines = 0
    ok_lines = 0
    #logs, lastlogs = {}, None
    #iter = 0
    for step, fulldata in enumerate(dataLoader):
        batchData, labelData = fulldata
        speakerLabel, groundTruth = labelData  # first one is speaker; groundTruth are 1-hot
        # n_examples += batchData.size(0)

        batchData, groundTruth, ok, bad = removeInvalidClassifData(batchData, groundTruth)
        bad_lines += bad
        ok_lines += ok

        if ok_lines == 0:
            continue

        # those detaches are not really needed
        batchData = batchData.detach().cuda(non_blocking=True)
        speakerLabel = speakerLabel.detach().cuda(non_blocking=True)
        groundTruth = groundTruth.detach().cuda(non_blocking=True)

        # [!] double checking stuff is detached and won't affect unsupervised training
        with torch.no_grad():
            #_, encoded_data, _ = cpcModel(batchData, speakerLabel)
            encoded_data = encoderNet(batchData).permute(0, 2, 1)
        encoded_data = encoded_data.detach().requires_grad_()

        modelGuess = classifModel(encoded_data)
        #      downsampling x160 made directly in the dataset with the labels
        #      could also do it here, but would be slow and big tenstors
        #      could also pass downsampling factor to DS (and use in getDataLoader(), loadNextPack()), but not needed for now
        # encoderDownsampling = cpcModel.module.gEncoder.DOWNSAMPLING
        
        flattenedGuess = modelGuess.view(-1, modelGuess.shape[2])
        flattenedTruth = groundTruth.view(-1)
        assert torch.max(flattenedTruth) < flattenedGuess.shape[1] and torch.min(flattenedTruth) >= 0
        classifLoss = classifCriterion(flattenedGuess, flattenedTruth)
        
        classifLoss.backward()

        # Show grads ?
        optimizer.step()
        optimizer.zero_grad()

        # if "locLoss_train" not in logs:
        #     logs["locLoss_train"] = np.zeros(allLosses.size(1))
        #     logs["locAcc_train"] = np.zeros(allLosses.size(1))

        # iter += 1
        # logs["locLoss_train"] += (allLosses.mean(dim=0)).detach().cpu().numpy()
        # logs["locAcc_train"] += (allAcc.mean(dim=0)).cpu().numpy()

        # if (step + 1) % loggingStep == 0:
        #     new_time = time.perf_counter()
        #     elapsed = new_time - start_time
        #     print(f"Update {step + 1}")
        #     print(f"elapsed: {elapsed:.1f} s")
        #     print(
        #         f"{1000.0 * elapsed / loggingStep:.1f} ms per batch, {1000.0 * elapsed / n_examples:.1f} ms / example")
        #     locLogs = utils.update_logs(logs, loggingStep, lastlogs)
        #     lastlogs = deepcopy(logs)
        #     utils.show_logs("Training loss", locLogs)
        #     start_time, n_examples = new_time, 0

    print(f'Training lines with ok label data: {ok_lines}, with missing data: {bad_lines}')
    # if scheduler is not None:
    #     scheduler.step()

    # logs = utils.update_logs(logs, iter)
    # logs["iter"] = iter
    # utils.show_logs("Average training loss on epoch", logs)
    #return logs


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

        batchData, label = fulldata

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


def valClassifStep(dataLoader,
              encoderNet,
              classifModel): #,
              #scheduler,
              #loggingStep):

    #cpcModel.eval()
    encoderNet.eval()
    classifModel.eval()
    
    #start_time = time.perf_counter()
    n_examples = 0
    correct = 0
    bad_lines = 0
    ok_lines = 0
    #logs, lastlogs = {}, None
    #iter = 0
    for step, fulldata in enumerate(dataLoader):
        batchData, labelData = fulldata
        speakerLabel, groundTruth = labelData  # first one is speaker; groundTruth are 1-hot
        
        batchData, groundTruth, ok, bad = removeInvalidClassifData(batchData, groundTruth)
        bad_lines += bad
        ok_lines += ok

        if ok_lines == 0:
            continue

        with torch.no_grad():
            batchData = batchData.cuda(non_blocking=True)
            speakerLabel = speakerLabel.cuda(non_blocking=True)
            groundTruth = groundTruth.cuda(non_blocking=True)
            #_, encoded_data, _ = cpcModel(batchData, speakerLabel)
            encoded_data = encoderNet(batchData).permute(0, 2, 1)
            modelGuess = classifModel(encoded_data)
            modelGuessFlattened = modelGuess.view(-1, modelGuess.shape[2])
            groundTruthFlattened = groundTruth.view(-1)
            modelLabels = torch.max(modelGuessFlattened, 1).indices
            # ground truth labels are already downsampled 160x
            #groundTruthLabels = torch.max(groundTruthFlattened, 1)  # now already as labels 
            n_examples += groundTruthFlattened.shape[0]
            assert torch.max(groundTruthFlattened) < modelGuessFlattened.shape[1] and torch.min(groundTruthFlattened) >= 0
            correct += (modelLabels == groundTruthFlattened).sum().item()

    print(f'Validation lines with ok label data: {ok_lines}, with missing data: {bad_lines}')

    return float(correct) / float(n_examples)        


def captureStep(
            dataLoader,
            cpcModel,
            cpcCriterion,
            captureOptions,
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
    if 'align' in whatToSave:
        cpcCaptureOpts.append('align')

    # they merge (perhaps each speaker's) audio into one long chunk
    # and AFAIU sample can begin in one file and end in other one
    # so won't try to mess up with tracking filenames, saving samples just as 1, 2, etc.

    batchBegin = 0
    epochDir = os.path.join(capturePath, str(epochNr))
    if not os.path.exists(epochDir):
        os.makedirs(epochDir)

    for step, fulldata in enumerate(dataLoader):

        batchData, label = fulldata
        batchEnd = batchBegin + batchData.shape[0] - 1

        batchData = batchData.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with torch.no_grad():

            c_feature, encoded_data, label = cpcModel(batchData, label)
            allLosses, allAcc, captured = cpcCriterion(c_feature, encoded_data, label, cpcCaptureOpts)
        
            # saving it with IDs like that assumes deterministic order of elements
            # which is there as dataLoader is a sequential one here
            if 'repr' in whatToSave:
                # encoded data shape: batch_size x len x repr_dim
                torch.save(encoded_data.cpu(), os.path.join(epochDir, f'repr_batch{batchBegin}-{batchEnd}.pt'))
            for cpcCaptureThing in cpcCaptureOpts:
                # pred shape (CPC-CTC): batch_size x (len - num_matched) x repr_dim x num_predicts
                # align shape (CPC-CTC): batch_size x (len - num_matched) x num_matched
                torch.save(captured[cpcCaptureThing].cpu(), os.path.join(epochDir, f'{cpcCaptureThing}_batch{batchBegin}-{batchEnd}.pt'))

            # TODO maybe later can write that with process pool or something??? but not even sure if makes sense

        batchBegin += batchData.shape[0]

    return


def classifTraining(
              getTrainLoader,
              getValLoader,
              encoderNet,
              classifModel,
              classifCriterion,
              optimizer,
              numEpochs,
              fileToWrite=None):
    
    print("--< COMPUTING SUPERVISED PHONEME CLASSIFICATION METRIC VIA TRAINING ON REPRESENTATIONS >--")

    valAccs = [0.]
    for classifEpoch in range(numEpochs):

        print(f"Starting classification task epoch {classifEpoch}")

        trainLoader = getTrainLoader()
        valLoader = getValLoader()

        # they implemented CPC in a way that does make comuting only representations difficult,
        # would need to change quite a bit of code, but maybe can stay like that
        
        trainClassifStep(
            trainLoader,
            encoderNet,
            classifModel,
            classifCriterion,
            optimizer
        )

        print("train step done")

        valLossThisEpoch = valClassifStep(
            valLoader,
            encoderNet,
            classifModel
        )

        print("val step done")

        if not math.isnan(valLossThisEpoch):
            valAccs.append(valLossThisEpoch)

        print(f"Phoneme classification accuracy in this epoch: {valLossThisEpoch}")

        if fileToWrite:
            with open(fileToWrite, 'a') as f:
                f.write(f'Val accuracy for epoch {classifEpoch}: {valLossThisEpoch}\n')

    if fileToWrite:
        with open(fileToWrite, 'a') as f:
            f.write(f'Best validation set accuracy obtained: {max(valAccs)}\n')
    return max(valAccs), valAccs


def run(performTraining,
        trainDataset,
        valDataset,
        captureDatasetWithOptions,
        batchSize,
        samplingMode,
        cpcModel,
        cpcCriterion,
        nEpoch,
        pathCheckpoint,
        optimizer,
        scheduler,
        logs,
        classifTaskParams=None):

    startEpoch = len(logs["epoch"])
    if performTraining:
        print(f"Running {nEpoch} epochs")
        bestAcc = 0
        bestStateDict = None
        start_time = time.time()
    captureDataset, captureOptions = captureDatasetWithOptions
    assert (captureDataset is None and captureOptions is None) \
        or (captureDataset is not None and captureOptions is not None)
    if captureOptions is not None:
        captureEachEpochs = captureOptions['eachEpochs']
    print(f'DS sizes: train {str(len(trainDataset)) if trainDataset is not None else "-"}, '
        f'val {str(len(valDataset)) if valDataset is not None else "-"}, capture '
        f'{str(len(captureDataset)) if captureDataset is not None else "-"}')
    if classifTaskParams:

        # will create a new model for each classif training instead of messing with reset
        encoderNet, trainClassifDataset, valClassifDataset, classifBatch, \
        classifEachEpochs, classifTrainEpochs, classifModelCreate, \
            only_classif_metric_output = classifTaskParams

        if 'phonemeClassif' not in logs:
            logs['phonemeClassif'] = []
        while len(logs['phonemeClassif']) < startEpoch:
            logs['phonemeClassif'].append(None)

    if performTraining:
        for epoch in range(startEpoch, nEpoch):

            print(f"Starting epoch {epoch}")
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
                captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, epoch)

            if classifTaskParams is not None and epoch % classifEachEpochs == 0:
                classifModel, classifCriterion, classifOptimizer =  classifModelCreate()
                bestClassifValAcc, _ = classifTraining(
                    (lambda ds, bs: 
                        (lambda: ds.getDataLoader(bs, "uniform", True, numWorkers=0))
                      )(trainClassifDataset, classifBatch),
                    (lambda ds, bs: 
                        (lambda: ds.getDataLoader(bs, 'sequential', False, numWorkers=0))
                      )(valClassifDataset, classifBatch),
                    encoderNet,
                    classifModel,
                    classifCriterion,
                    classifOptimizer,
                    classifTrainEpochs
                )
                # freeing stuff just in case
                del classifModel
                del classifCriterion
                del classifOptimizer
                logs['phonemeClassif'].append(bestClassifValAcc)
                print(f'LEARNT phoneme classification with accuracy: {bestClassifValAcc}')
            elif classifTaskParams is not None:
                logs['phonemeClassif'].append(None)

            print(f'Ran {epoch + 1} epochs '
                f'in {time.time() - start_time:.2f} seconds')

            torch.cuda.empty_cache()

            currentAccuracy = float(locLogsVal["locAcc_val"].mean())
            if currentAccuracy > bestAcc:
                bestStateDict = fl.get_module(cpcModel).state_dict()

            for key, value in dict(locLogsTrain, **locLogsVal).items():
                if key not in logs:
                    logs[key] = [None for x in range(epoch)]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
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
    else:

        didSomething = False
        if captureDataset is not None:
            assert captureDataset is not None and captureOptions is not None
            
            # here we ignore num epochs, epoch frequency to log etc. - just capturing data
            # for the model training saved in the provided checkpoint
            
            captureLoader = captureDataset.getDataLoader(batchSize, 'sequential', False,
                                                    numWorkers=0)
            print(f"Capturing data for epoch {startEpoch}")
            captureStep(captureLoader, cpcModel, cpcCriterion, captureOptions, startEpoch)
            didSomething = True

        if classifTaskParams and only_classif_metric_output is not None:
            classifModel, classifCriterion, classifOptimizer =  classifModelCreate()
            with open(only_classif_metric_output, 'w') as f:
                f.write("")  # clearing the file
            bestClassifValAcc, allClassifAccs = classifTraining(
                (lambda ds, bs: 
                    (lambda: ds.getDataLoader(bs, "uniform", True, numWorkers=0))
                    )(trainClassifDataset, classifBatch),
                (lambda ds, bs: 
                    (lambda: ds.getDataLoader(bs, 'sequential', False, numWorkers=0))
                    )(valClassifDataset, classifBatch),
                encoderNet,
                classifModel,
                classifCriterion,
                classifOptimizer,
                classifTrainEpochs,
                fileToWrite=only_classif_metric_output
            )
            # freeing stuff just in case
            del classifModel
            del classifCriterion
            del classifOptimizer
            
            didSomething = True

        assert didSomething



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
    json.dump(vars(args), open(os.path.join(args.pathCheckpoint, 'checkpoint_args.json'), 'wt'))
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

    if not args.onlyCapture:
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
        for argVal, name in zip([args.saveRepr, args.savePred, args.saveAlign], ['repr', 'pred', 'align']):
            if argVal:
                whatToSave.append(name)
        assert len(whatToSave) > 0
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

        if args.compute_supervised_phoneme_classif_metric:
            assert args.pathTrain is not None and args.pathVal is not None
            assert args.pathTrainAlignments is not None and args.pathValAlignments is not None
            pathTrainAlignments = args.pathTrainAlignments
            pathValAlignments = args.pathValAlignments
            print("")
            print(f'Loading audio data at {args.pathDB}')
            print("Loading the classif training dataset")
            trainClassifDataset = AudioBatchData(args.pathDB,
                                        args.sizeWindow,
                                        seqTrain,
                                        phoneLabels,
                                        len(speakers),
                                        nProcessLoader=args.n_process_loader,
                                        MAX_SIZE_LOADED=args.max_size_loaded,
                                        pathAlignments=pathTrainAlignments)
            print("Classif training dataset loaded")
            print("")

            print("Loading the classif validation dataset")
            valClassifDataset = AudioBatchData(args.pathDB,
                                        args.sizeWindow,
                                        seqVal,
                                        phoneLabels,
                                        len(speakers),
                                        nProcessLoader=args.n_process_loader,
                                        pathAlignments=pathValAlignments)
            print("Classif validation dataset loaded")
            print("")

            trainPhonemes = trainClassifDataset.getPhoneDict()
            valPhonemes = valClassifDataset.getPhoneDict()
            mergedDict = {}
            for d in (trainPhonemes, valPhonemes):
                for p in d:
                    if p not in mergedDict:
                        mergedDict[p] = len(mergedDict)
            trainClassifDataset.setPhoneDict(deepcopy(mergedDict))
            valClassifDataset.setPhoneDict(deepcopy(mergedDict))


        else:
            phoneLabels, nPhones = None, None
        
    else:
        phoneLabels, nPhones = None, None
        trainDataset = None
        valDataset = None

    if seqCapture is not None:
        print("Loading the capture dataset")
        captureDataset = AudioBatchData(args.pathDB,
                                    args.sizeWindow,
                                    seqCapture,
                                    phoneLabels,
                                    len(speakers),
                                    nProcessLoader=args.n_process_loader)
        print("Capture dataset loaded")
        print("")
    else:
        captureDataset = None

    if args.load is not None:
        # AFAIU args.hiddenEncoder is encoded representation dimension, from single or merged-model encoder
        # AR stuff AFAIU is the CPC prediction net
        cpcModel, args.hiddenGar, args.hiddenEncoder = \
            fl.loadModel(args.load)
        encodedDim = args.hiddenEncoder

    else:
        # Encoder network
        encoderNet = fl.getEncoder(args)
        
        # AR Network
        arNet = fl.getAR(args)
        encodedDim = encoderNet.getDimOutput()

        cpcModel = model.CPCModel(encoderNet, arNet)

    if args.compute_supervised_phoneme_classif_metric:
        
        
        def classifModelCreateFun():   # TODO maybe bind args better to be sure
            layers = []
            neuronNumbers = list(map(int, args.FCNetLayersNeurons.split(',')))
            prevDim = encodedDim
            for nNum in neuronNumbers:
                if nNum != 0:  # can put 0 to specify e.g. no hidden layers
                    layers.append(torch.nn.Linear(prevDim, nNum))
                    prevDim = nNum
                    layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(prevDim, len(mergedDict)))

            classifModel = torch.nn.Sequential(*layers)
            classifCriterion = torch.nn.CrossEntropyLoss()
            classifOptimizer = torch.optim.Adam(classifModel.parameters(), lr=args.classif_lr)
        
            classifCriterion.cuda()
            classifModel.cuda()
            return (classifModel, classifCriterion, classifOptimizer)

        classifParams = (cpcModel.gEncoder, trainClassifDataset, valClassifDataset, args.classifBatch,
                         args.classifEachEpochs, args.classifTrainEpochs, classifModelCreateFun,
                         args.only_classif_metric_output if args.only_classif_metric_output else None)

    else:
        classifParams = None
        
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
    optimizer = torch.optim.AdamW(g_params, lr=lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01)
                                 #betas=(args.beta1, args.beta2),
                                 #eps=args.epsilon)

    if loadOptimizer:
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
    print("cpcCriterion", cpcCriterion)

    cpcModel = torch.nn.DataParallel(cpcModel,
                                     device_ids=range(args.nGPU)).cuda()
    cpcCriterion = torch.nn.DataParallel(cpcCriterion,
                                         device_ids=range(args.nGPU)).cuda()
    
    run(True if (not args.onlyCapture) and (args.only_classif_metric_output is None) else False,
        trainDataset,
        valDataset,
        (captureDataset, captureOptions),
        batchSize,
        args.samplingType,
        cpcModel,
        cpcCriterion,
        args.nEpoch,
        args.pathCheckpoint,
        optimizer,
        scheduler,
        logs,
        classifTaskParams=classifParams)


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
                          help='total number of data points to capture data for; conflicts with --captureDSfreq')
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
    # [!] --pathPhone and whole this group_supervised is for some their weird format, 
    #     I looked at the code and it seemed so absurdly unnecessary complicated and weird
    #     that I just decided to base additional supervised (metrics only)
    #     on alignments in format I know - this is done in group_supervised_metric below
    group_supervised.add_argument('--pathPhone', type=str, default=None,
                                  help='(Supervised mode only) Path to a .txt '
                                  'containing the phone labels of the dataset. If given '
                                  'and --supervised, will train the model using a '
                                  'phone classification task.')
    group_supervised.add_argument('--CTC', action='store_true')

    group_supervised_metric = parser.add_argument_group(
                          'Supervised metrics - to see how CTC performs, see how well it performs in '
                          'additional supervised phoneme classification task (with fully connected net)')
    group_supervised_metric.add_argument('--compute_supervised_phoneme_classif_metric',
                          action='store_true', help='compute the metric; conflicts with --onlyCapture '
                          'except if --only_classif_metric_output')
    group_supervised_metric.add_argument('--only_classif_metric_output',
                          type=str, default=None,
                          help="don't train CPC, just compute classification accuracy on given checkpoint "
                          'on given checkpoint (classification net itself is trained) and store in given path; '
                          'conflicts with regular training')
    group_supervised_metric.add_argument('--pathTrainAlignments', type=str, default=None,
                          help='Path to a root directory with alinment files '
                          'in savage Praat TextGrid format; superset subdir structure to dataset assumed '
                          '- needs to contain sub-paths to alignment files for all files in train DS '
                          'can contains more - e.g. can pass same root when using two subsets of that')
    group_supervised_metric.add_argument('--pathValAlignments', type=str, default=None,
                          help='Path to a root directory with alinment files '
                          'in savage Praat TextGrid format; superset subdir structure to dataset assumed '
                          '- needs to contain sub-paths to alignment files for all files in val DS '
                          'can contains more - e.g. can pass same root when using two subsets of that')
    group_supervised_metric.add_argument('--classifEachEpochs', type=int, default=20,
                          help='how often to perform classification task - classification net is then '
                          'trained on train DS representations and assesed on val DS representations '
                          'that are produced after that epoch in eval mode')
    group_supervised_metric.add_argument('--FCNetLayersNeurons', type=str, default='1000,1000',
                          help='description of how big net to use for classification in ,-separated format '
                          '- e.g. 1000,1000 will result in a net with 2 hidden layers of 1000 neurons; ' 
                          ' additionally softmax layer with correct number of classes is added on top')
    group_supervised_metric.add_argument('--classif_lr', type=float, default=0.0001,
                          help='what lr is used for the classification net (for adam)')
    group_supervised_metric.add_argument('--classifTrainEpochs', type=int, default=30,
                          help='how many epochs to perform during classification model training')
    group_supervised_metric.add_argument('--classifBatch', type=int, default=16,
                          help='how big batch to use for classification task')

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
    group_save.add_argument('--saveRepr', action='store_true', help='if to save representations after the encoder')
    group_save.add_argument('--savePred', action='store_true', help='if to save CPC predictions')
    group_save.add_argument('--saveAlign', action='store_true', help='if to save CTC alignments with CPC predictions - only for CPC-CTC variant')
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
