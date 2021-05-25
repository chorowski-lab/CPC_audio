# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
import os

import cpc.criterion as cr
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
from cpc.model import CPCModelNullspace


def train_step(feature_maker, centerModel, segmentCostModel, criterion, data_loader, optimizer, label_key, cpc_epochs):

    if feature_maker.optimize:
        feature_maker.train()
    criterion.train()

    logs = {"locLoss_train": 0,  "locAcc_train": 0}

    cpcEpochNr, cpcAllEpochs = cpc_epochs
    
    for step, fulldata in enumerate(data_loader):

        optimizer.zero_grad()
        batch_data, label_data = fulldata
        label = label_data[label_key]
        givenCenters = centerModel.centersForStuff(cpcEpochNr) if centerModel is not None else None
        numGPUs = len(feature_maker.device_ids)
        if givenCenters is not None:
            givenCenters = givenCenters.repeat(numGPUs,1)
        if segmentCostModel is not None:
            maxAllowedSegmCost = segmentCostModel.getCurrentMaxCostEstimator()
        else:
            maxAllowedSegmCost = None
        c_feature, encoded_data, _, _, _, _, _, _, _ = feature_maker(batch_data, None, None, maxAllowedSegmCost, givenCenters, cpc_epochs, False, False)
        if not feature_maker.optimize:
            c_feature, encoded_data = c_feature.detach(), encoded_data.detach()
            # ("TODO") center model not updated in this case and it should but it's unused

        all_losses, all_acc = criterion(c_feature, encoded_data, label)

        totLoss = all_losses.sum()
        totLoss.backward()
        optimizer.step()

        logs["locLoss_train"] += np.asarray([all_losses.mean().item()])
        logs["locAcc_train"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, step)
    logs["iter"] = step

    return logs


def val_step(feature_maker, centerModel, segmentCostModel, criterion, data_loader, label_key, cpc_epochs):

    feature_maker.eval()
    criterion.eval()
    logs = {"locLoss_val": 0,  "locAcc_val": 0}

    cpcEpochNr, cpcAllEpochs = cpc_epochs

    for step, fulldata in enumerate(data_loader):

        with torch.no_grad():
            batch_data, label_data = fulldata
            label = label_data[label_key]
            givenCenters = centerModel.centersForStuff(cpcEpochNr) if centerModel is not None else None
            numGPUs = len(feature_maker.device_ids)
            if givenCenters is not None:
                givenCenters = givenCenters.repeat(numGPUs,1)
            if segmentCostModel is not None:
                maxAllowedSegmCost = segmentCostModel.getCurrentMaxCostEstimator()
            else:
                maxAllowedSegmCost = None
            c_feature, encoded_data, _, _, _, _, _, _, _ = feature_maker(batch_data, None, None, maxAllowedSegmCost, givenCenters, cpc_epochs, False, False)
            all_losses, all_acc = criterion(c_feature, encoded_data, label)

            logs["locLoss_val"] += np.asarray([all_losses.mean().item()])
            logs["locAcc_val"] += np.asarray([all_acc.mean().item()])

    logs = utils.update_logs(logs, step)

    return logs

# TODO centerModel & segmentCostModel update
def run(feature_maker,
        criterion,
        train_loader,
        val_loader,
        optimizer,
        logs,
        n_epochs,
        path_checkpoint,
        label_key,
        cpc_epochs=(100,100)):


    start_epoch = len(logs["epoch"])
    best_acc = -1

    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):

        sys.stdout.flush()

        logs_train = train_step(feature_maker, criterion, train_loader,
                                optimizer, label_key, cpc_epochs)
        logs_val = val_step(feature_maker, criterion, val_loader, label_key, cpc_epochs)

        print('')
        print('_'*50)
        print(f'Ran {epoch + 1} epochs '
              f'in {time.time() - start_time:.2f} seconds')
        utils.show_logs("Training loss", logs_train)
        utils.show_logs("Validation loss", logs_val)
        print('_'*50)
        print('')

        if logs_val["locAcc_val"] > best_acc:
            best_state = deepcopy(fl.get_module(feature_maker).state_dict())
            best_acc = logs_val["locAcc_val"]

        logs["epoch"].append(epoch)
        for key, value in dict(logs_train, **logs_val).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        if (epoch % logs["saveStep"] == 0 and epoch > 0) or epoch == n_epochs - 1:
            model_state_dict = fl.get_module(feature_maker).state_dict()
            criterion_state_dict = fl.get_module(criterion).state_dict()

            fl.save_checkpoint(model_state_dict, criterion_state_dict,
                               optimizer.state_dict(), best_state, None,
                               f"{path_checkpoint}_{epoch}.pt")
            utils.save_logs(logs, f"{path_checkpoint}_logs.json")


def save_linsep_best_checkpoint(cpc_model_state, classif_net_criterion_state, optimizer_state, 
                    path_checkpoint):

    state_dict = {"CPCmodel": cpc_model_state,
                  "classifNetCriterionCombined": classif_net_criterion_state,
                  "optimizer": optimizer_state}

    torch.save(state_dict, path_checkpoint)

def trainLinsepClassification(
        feature_maker,
        centerModel,
        segmentCostModel,
        criterion,  # combined with classification model before
        train_loader,
        val_loader,
        optimizer,
        path_logs,
        logs_save_step,
        path_best_checkpoint,
        n_epochs,
        cpc_epochs,
        label_key):

    wasOptimizeCPC = feature_maker.optimize if hasattr(feature_maker, 'optimize') else None
    feature_maker.eval()
    feature_maker.optimize = False

    current_cpc_epoch, all_cpc_epochs = cpc_epochs

    start_epoch = 0
    best_train_acc = -1
    best_acc = -1
    bect_epoch = -1
    logs = {"epoch": [], "iter": [], "saveStep": logs_save_step}

    start_time = time.time()

    for epoch in range(start_epoch, n_epochs):

        sys.stdout.flush()

        logs_train = train_step(feature_maker, centerModel, segmentCostModel, criterion, train_loader,
                                optimizer, label_key, cpc_epochs)
        logs_val = val_step(feature_maker, centerModel, segmentCostModel, criterion, val_loader, label_key, cpc_epochs)

        print('')
        print('_'*50)
        print(f'Ran {epoch + 1} {label_key} classification epochs '
              f'in {time.time() - start_time:.2f} seconds')
        utils.show_logs("Training loss", logs_train)
        utils.show_logs("Validation loss", logs_val)
        print('_'*50)
        print('')

        if logs_val["locAcc_val"] > best_acc:
            best_state_cpc = deepcopy(fl.get_module(feature_maker).state_dict())
            best_state_classif_crit = deepcopy(fl.get_module(criterion).state_dict())
            optimizer_state_best_ep = optimizer.state_dict()
            best_epoch = epoch
            best_acc = logs_val["locAcc_val"]

        if logs_train["locAcc_train"] > best_train_acc:
            best_train_acc = logs_train["locAcc_train"]

        logs["epoch"].append(epoch)
        for key, value in dict(logs_train, **logs_val).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        if (epoch % logs["saveStep"] == 0 and epoch > 0) or epoch == n_epochs - 1:
            model_state_dict = fl.get_module(feature_maker).state_dict()
            criterion_state_dict = fl.get_module(criterion).state_dict()

            # fl.save_checkpoint(model_state_dict, criterion_state_dict,
            #                    optimizer.state_dict(), best_state,
            #                    f"{path_checkpoint}_{epoch}.pt")
            utils.save_logs(logs, f"{path_logs}_logs.json")

    if path_best_checkpoint:
        save_linsep_best_checkpoint(best_state_cpc, best_state_classif_crit,
                        optimizer_state_best_ep,  # TODO check if should save that epoch or last in optimizer
                        os.path.join(path_best_checkpoint, f"{label_key}_classif_best-epoch{best_epoch}-cpc_epoch{current_cpc_epoch}.pt"))
    feature_maker.optimize = wasOptimizeCPC
    return {'num_epoch_trained': n_epochs,
            'best_val_acc': best_acc,
            'best_train_acc': best_train_acc
            }


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Linear separability trainer'
                                     ' (default test in speaker separability)')
    parser.add_argument('pathDB', type=str,
                        help="Path to the directory containing the audio data.")
    parser.add_argument('pathTrain', type=str,
                        help="Path to the list of the training sequences.")
    parser.add_argument('pathVal', type=str,
                        help="Path to the list of the test sequences.")
    parser.add_argument('load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels. If given, will"
                        " compute the phone separability.")
    parser.add_argument('--CTC', action='store_true',
                        help="Use the CTC loss (for phone separability only)")
    parser.add_argument('--pathCheckpoint', type=str, default='out',
                        help="Path of the output directory where the "
                        " checkpoints should be dumped.")
    parser.add_argument('--nGPU', type=int, default=-1,
                        help='Bumber of GPU. Default=-1, use all available '
                        'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--unfrozen', action='store_true',
                        help="If activated, update the feature network as well"
                        " as the linear classifier")
    parser.add_argument('--no_pretraining', action='store_true',
                        help="If activated, work from an untrained model.")
    parser.add_argument('--file_extension', type=str, default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--save_step', type=int, default=-1,
                        help="Frequency at which a checkpoint should be saved,"
                        " et to -1 (default) to save only the best checkpoint.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Value of beta1 for the Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Value of beta2 for the Adam optimizer.')
    parser.add_argument('--epsilon', type=float, default=2e-8,
                        help='Value of epsilon for the Adam optimizer.')
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")
    parser.add_argument('--size_window', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    parser.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    parser.add_argument("--model", type=str, default="cpc",
                          help="Pre-trained model architecture ('cpc' [default] or 'wav2vec2').")
    parser.add_argument("--path_fairseq", type=str, default="/pio/scratch/1/i273233/fairseq",
                          help="Path to the root of fairseq repo.")
    parser.add_argument("--mode", type=str, default="phonemes",
                          help="Mode for example phonemes, speakers, speakers_factorized, phonemes_nullspace")
    parser.add_argument("--path_speakers_factorized", type=str, default="/pio/scratch/1/i273233/linear_separability/cpc/cpc_official_speakers_factorized/checkpoint_9.pt",
                          help="Path to the checkpoint from speakers factorized")
    parser.add_argument('--dim_inter', type=int, default=128, help="Dimension between factorized matrices (dim_features x dim_inter) x (dim_inter x len(speakers)) ")
    parser.add_argument('--gru_level', type=int, default=-1,
                        help='Hidden level of the LSTM autoregressive model to be taken'
                        '(default: -1, last layer).')

    parser.add_argument('--centerpushFile', type=str, default=None, help="path to checkpoint containing cluster centers")
    parser.add_argument('--centerpushDeg', type=float, default=None, help="part of (euclidean) distance to push to the center")

    args = parser.parse_args(argv)
    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()
    if args.save_step <= 0:
        args.save_step = args.n_epoch

    args.load = [str(Path(x).resolve()) for x in args.load]
    args.pathCheckpoint = str(Path(args.pathCheckpoint).resolve())

    return args


def main(argv):

    args = parse_args(argv)
    logs = {"epoch": [], "iter": [], "saveStep": args.save_step}
    load_criterion = False

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache)

    if args.model == "cpc":
        def loadCPCFeatureMaker(pathCheckpoint, gru_level=-1, get_encoded=False, keep_hidden=True):
            """
            Load CPC Feature Maker from CPC checkpoint file.
            """
            # Set LSTM level
            if gru_level is not None and gru_level > 0:
                updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
            else:
                updateConfig = None

            # Load CPC model
            model, nHiddenGar, nHiddenEncoder = fl.loadModel(pathCheckpoint, args.batchSizeGPU, updateConfig=updateConfig)
            
            # Keep hidden units at LSTM layers on sequential batches
            model.gAR.keepHidden = keep_hidden

            # Build CPC Feature Maker from CPC model
            #featureMaker = fl.FeatureModule(model, get_encoded=get_encoded)

            #return featureMaker
            return model, nHiddenGar, nHiddenEncoder

        if args.gru_level is not None and args.gru_level > 0:
            model, hidden_gar, hidden_encoder = loadCPCFeatureMaker(args.load, gru_level=args.gru_level)
        else:
            model, hidden_gar, hidden_encoder = fl.loadModel(args.load, args.batchSizeGPU,
                                                     loadStateDict=not args.no_pretraining)

        dim_features = hidden_encoder if args.get_encoded else hidden_gar
    else:
        sys.path.append(os.path.abspath(args.path_fairseq))
        from fairseq import checkpoint_utils

        def loadCheckpoint(path_checkpoint, path_data):
            """
            Load lstm_lm model from checkpoint.
            """
            # Set up the args Namespace
            model_args = argparse.Namespace(
                task="language_modeling",
                output_dictionary_size=-1,
                data=path_data,
                path=path_checkpoint
                )
            
            # Load model
            models, _model_args = checkpoint_utils.load_model_ensemble([model_args.path])
            model = models[0]
            return model

        model = loadCheckpoint(args.load[0], args.pathDB)
        dim_features = 768

    dim_inter = args.dim_inter
    # Now the criterion

    if args.mode == "phonemes_nullspace" or args.mode == "speakers_nullspace":
        speakers_factorized = cr.SpeakerDoubleCriterion(dim_features, dim_inter, len(speakers))
        speakers_factorized.load_state_dict(torch.load(args.path_speakers_factorized)["cpcCriterion"])
        for param in speakers_factorized.parameters():
            param.requires_grad = False

        def my_nullspace(At, rcond=None):
            ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
            vht=vht.T        
            Mt, Nt = ut.shape[0], vht.shape[1] 
            if rcond is None:
                rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
            tolt = torch.max(st) * rcondt
            numt= torch.sum(st > tolt, dtype=int)
            nullspace = vht[numt:,:].T.cpu().conj()
            # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
            return nullspace

        dim_features = dim_features - dim_inter
        nullspace = my_nullspace(speakers_factorized.linearSpeakerClassifier[0].weight)
        model = CPCModelNullspace(model, nullspace)

    phone_labels = None
    if args.pathPhone is not None:

        phone_labels, n_phones = parseSeqLabels(args.pathPhone)
        label_key = 'phone'

        if not args.CTC:
            print(f"Running phone separability with aligned phones")
            criterion = cr.PhoneCriterion(dim_features,
                                          n_phones, args.get_encoded)
        else:
            print(f"Running phone separability with CTC loss")
            criterion = cr.CTCPhoneCriterion(dim_features,
                                             n_phones, args.get_encoded)
    else:
        label_key = 'speaker'
        print(f"Running speaker separability")
        if args.mode == "speakers_factorized":
            criterion = cr.SpeakerDoubleCriterion(dim_features, dim_inter, len(speakers))
        else:
            criterion = cr.SpeakerCriterion(dim_features, len(speakers))
    criterion.cuda()
    criterion = torch.nn.DataParallel(criterion, device_ids=range(args.nGPU))

    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    # Dataset
    seq_train = filterSeqs(args.pathTrain, seqNames)
    seq_val = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seq_train = seq_train[:1000]
        seq_val = seq_val[:100]

    db_train = AudioBatchData(args.pathDB, args.size_window, seq_train,
                              (phone_labels, n_phones), len(speakers), nProcessLoader=args.n_process_loader,
                                  MAX_SIZE_LOADED=args.max_size_loaded)
    db_val = AudioBatchData(args.pathDB, args.size_window, seq_val,
                            (phone_labels, n_phones), len(speakers), nProcessLoader=args.n_process_loader,
                            MAX_SIZE_LOADED=args.max_size_loaded)


    batch_size = args.batchSizeGPU * args.nGPU

    train_loader = db_train.getDataLoader(batch_size, "uniform", True,
                                          numWorkers=0)

    val_loader = db_val.getDataLoader(batch_size, 'sequential', False,
                                      numWorkers=0)

    # Optimizer
    g_params = list(criterion.parameters())
    model.optimize = False
    model.eval()
    if args.unfrozen:
        print("Working in full fine-tune mode")
        g_params += list(model.parameters())
        model.optimize = True
    else:
        print("Working with frozen features")
        for g in model.parameters():
            g.requires_grad = False

    optimizer = torch.optim.Adam(g_params, lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.epsilon)

    # Checkpoint directory
    args.pathCheckpoint = Path(args.pathCheckpoint)
    args.pathCheckpoint.mkdir(exist_ok=True)
    args.pathCheckpoint = str(args.pathCheckpoint / "checkpoint")

    with open(f"{args.pathCheckpoint}_args.json", 'w') as file:
        json.dump(vars(args), file, indent=2)

    if args.centerpushFile:
        clustersFileExt = args.centerpushFile.split('.')[-1]
        assert clustersFileExt in ('pt', 'npy', 'txt')
        if clustersFileExt == 'npy':
            centers = np.load(args.centerpushFile)
        elif clustersFileExt == 'txt':
            centers = np.genfromtxt(args.centerpushFile)
        elif clustersFileExt == 'pt':  # assuming it's a checkpoint
            centers = torch.load(args.centerpushFile, map_location=torch.device('cpu'))['state_dict']['Ck']
            centers = torch.reshape(centers, centers.shape[1:]).numpy()
        centers = torch.tensor(centers).cuda()
        centerpushSettings = (centers, args.centerpushDeg)
    else:
        centerpushSettings = None

    run(model, criterion, train_loader, val_loader, optimizer, logs,
        args.n_epoch, args.pathCheckpoint, label_key=label_key, centerpushSettings=centerpushSettings)



if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()
    
    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
