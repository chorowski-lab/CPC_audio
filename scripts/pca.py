#!/usr/bin/env python3 -u

import logging
import os
import sys
import argparse
from itertools import chain
from pathlib import Path
import time
import copy
import numpy as np
from sklearn.decomposition import IncrementalPCA



def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,
                        help="Path to the embeddings for ABX.")
    parser.add_argument("--path_output_dir", type=str,
                        help="Path to the output directory.")
    parser.add_argument("--n_components", type=int, default=None,
                        help="To how many dimensions do we want it reduced.")
    parser.add_argument("--file_extensions", type=list, default=["npz", "npy", "txt"],
                        help="Extensions of the embeddings (default: [npz, npy, txt]).")
    parser.add_argument("--no_test", action="store_true",
                        help="Don't compute embeddings for test-* parts of dataset")
    return parser.parse_args()

def main():
    # Parse and print args
    args = parse_args()
    pca = None

    # Extract values from chosen layers and save them to files
    phonetic = "phonetic"
    datasets_path = os.path.join(args.path_data, phonetic)
    datasets = os.listdir(datasets_path)
    print(datasets)

    bucket = None
    if pca is None:
        for dataset in datasets:
            print("> {}".format(dataset))
            dataset_path = os.path.join(datasets_path, dataset)
            files = [f for f in os.listdir(dataset_path) if any([f.endswith(ext) for ext in args.file_extensions])]
            for i, f in enumerate(files):
                print("Progress {:2.1%}".format(i / len(files)), end="\r")
                input_f = os.path.join(dataset_path, f)
                x = np.loadtxt(input_f)

                bucket = x if bucket is None else np.vstack((x, bucket))
                if bucket.shape[0] >= args.n_components:
                    if pca is None:
                        pca = IncrementalPCA(n_components=args.n_components)
                    pca.partial_fit(bucket)

                    bucket = None
                

    datasets = [dataset for dataset in datasets if not args.no_test or not dataset.startswith("test")]

    print("\nSaving...\n")
    for dataset in datasets:
        print("> {}".format(dataset))
        dataset_path = os.path.join(datasets_path, dataset)
        files = [f for f in os.listdir(dataset_path) if any([f.endswith(ext) for ext in args.file_extensions])]
        for i, f in enumerate(files):
            print("Progress {:2.1%}".format(i / len(files)), end="\r")
            input_f = os.path.join(dataset_path, f)
            x = np.loadtxt(input_f)
            x_pca = pca.transform(x)

            output_dir = os.path.join(args.path_output_dir, phonetic, dataset)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_f = os.path.join(output_dir, os.path.splitext(f)[0] + ".txt")
            np.savetxt(out_f, x_pca)
            

if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()
    main()

