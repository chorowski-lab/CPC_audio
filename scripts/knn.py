#!/usr/bin/env python3 -u

from functools import reduce
import logging
import os
from posixpath import join
import sys
import argparse
from itertools import chain, product
from pathlib import Path
import time
import copy
import numpy as np
import soundfile as sf

from cpc.feature_loader import loadModel, FeatureModule

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("zerospeech2021 abx")

def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,
                        help="Path to the dataset that we want to compute KNN for.")
    parser.add_argument('--path_train', type=str,
                        help="Path to the list of the training sequences.")
    parser.add_argument('--path_val', type=str,
                        help="Path to the list of the test sequences.")
    parser.add_argument('--path_phone', type=str, default=None,
                        help="Path to the phone labels. If given, will"
                        " compute the phone separability.")
    parser.add_argument("--path_output_dir", type=str,
                        help="Path to the output directory.")                     
    parser.add_argument("--path_checkpoint", type=str, default=None,
                          help="Load the checkpoint containing nullspace, if theres nothing provided, the script executes on the baseline embeddings")
    parser.add_argument('--nullspace', action='store_true',
                        help="Additionally load nullspace")
    parser.add_argument('--normalize', action='store_true',
                        help="Normalize each vector")
    parser.add_argument('--k', type=str, default=None,
                          help='K in K Nearest Neighbours that we want to use for classification. You can specify more than one K. Please separate them with a single comma')        
    parser.add_argument('--take_every_nth', type=int, default=10,
                          help='Take every <value> embedding from every training sequence')
    parser.add_argument("--file_extension", type=str, default="npy",
                          help="Extension of the audio embeddings in the dataset (default: npy).")  
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode")
    parser.add_argument('--scale_down', type=int, default=10,
                          help='Factor to scale down training set so that it isnt too big')
    parser.add_argument('--change_train_sample_period', type=int, default=100,
                          help='Switch to other sample of train set every change_train_sample_period')                          

    return parser.parse_args()

def main():
    # Parse and print args
    args = parse_args()
    logger.info(args)

    # Load the model
    print("")
    print(f"Loading model from {args.path_checkpoint}") 

    nullspace = None
    if args.path_checkpoint is not None:
        model = loadModel([args.path_checkpoint], load_nullspace=args.nullspace)[0]
        nullspace = model.nullspace.weight.detach().numpy().T

    phones = {line.split(" ")[0] : [int(phone) for phone in line.split(" ")[1:]] for line in open(args.path_phone)}
    unique_phones = {item for sublist in phones.values() for item in sublist}


    # Extract values from chosen layers and save them to files
    train_split = {file.strip() for file in open(args.path_train)}
    val_split = {file.strip() for file in open(args.path_val)}
    filenames = [file for root, dirs, files in os.walk(args.path_data) for file in files if file.endswith(args.file_extension)]
    train_filenames = [filename for filename in filenames if os.path.splitext(os.path.basename(filename))[0] in train_split]
    val_filenames = [filename for filename in filenames if os.path.splitext(os.path.basename(filename))[0] in val_split]

    if args.debug:
        train_filenames = train_filenames[:100]
        val_filenames = val_filenames[:100]

    embedding_dim = None
    train_set_size = 0
    
    cache = None
    cache_name = os.path.join(args.path_output_dir, "cached_size.npy")
    if os.path.exists(cache_name):
        print("Cache present")
        cache = np.load(cache_name)

    if cache is None:
        print("Computing train set size...")
        for i, f in enumerate(train_filenames):
            print("Progress {:2.1%}".format(i / len(train_filenames)), end="\r")
            input_f = os.path.join(args.path_data, f)
            x = np.load(input_f)
            labels = phones[os.path.splitext(os.path.basename(f))[0]]
            length = min(x.shape[0], len(labels))
            x = x[:length:args.take_every_nth, :]
            labels = labels[:length:args.take_every_nth]

            train_set_size += x.shape[0]
            if embedding_dim is None:
                if nullspace is not None:
                    x = x @ nullspace
                embedding_dim = x.shape[1]
        
        cache = (train_set_size, embedding_dim)
        Path(args.path_output_dir).mkdir(parents=True, exist_ok=True)
        np.save(cache_name, cache)
            
    train_set_size, embedding_dim = cache
    print(f"Train set size: {train_set_size}, Embedding size: {embedding_dim}")
    print("Loading train set...")
    train_set = np.empty((train_set_size, embedding_dim))
    train_set_labels = np.empty(train_set_size, dtype=np.int)
    counter = 0
    for i, f in enumerate(train_filenames):
        print("Progress {:2.1%}".format(i / len(train_filenames)), end="\r")
        input_f = os.path.join(args.path_data, f)
        x = np.load(input_f)
        labels = phones[os.path.splitext(os.path.basename(f))[0]]
        length = min(x.shape[0], len(labels))
        x = x[:length:args.take_every_nth, :]
        labels = labels[:length:args.take_every_nth]

        if nullspace is not None:
            x = x @ nullspace

        if args.normalize:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        train_set[counter : counter + x.shape[0], :] = x
        train_set_labels[counter : counter + x.shape[0]] = labels

        counter += x.shape[0]

    val_acuraccies = dict()
    ks = [int(k) for k in args.k.split(",")]
    for k in ks:
        print(f"KNN classifier for k = {k}...")
        knn = KNeighborsClassifier(k, n_jobs=20)
        val_results = []
        val_sizes = []

        for i, f in enumerate(val_filenames):
            if i % args.change_train_sample_period == 0:
                perm = np.random.choice(train_set.shape[0], train_set.shape[0] // args.scale_down, replace=False)
                train_set_sample = train_set[perm]
                train_set_labels_sample = train_set_labels[perm]
                knn.fit(train_set_sample, train_set_labels_sample)

            print("Progress {:2.1%}".format(i / len(val_filenames)), end="\r")
            input_f = os.path.join(args.path_data, f)
            x = np.load(input_f)
            labels = phones[os.path.splitext(os.path.basename(f))[0]]
            length = min(x.shape[0], len(labels))
            x = x[:length:args.take_every_nth, :]
            labels = labels[:length:args.take_every_nth]
            
            if nullspace is not None:
                x = x @ nullspace

            if args.normalize:
                x = x / np.linalg.norm(x, axis=1, keepdims=True)

            val_results.append(knn.score(x, labels))
            val_sizes.append(x.shape[0])
        
        val_acuraccy = sum([result * size for result, size in zip(val_results, val_sizes)]) / sum(val_sizes)
        val_acuraccies[k] = val_acuraccy

        Path(args.path_output_dir).mkdir(parents=True, exist_ok=True)
        output_file = "results.txt"
        with open(os.path.join(args.path_output_dir, output_file), 'a+') as file:
            print(f"{k} {val_acuraccy}", file=file)

        print(f"Validation accuracy on {k} is {val_acuraccy}")

            

if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()
    main()

