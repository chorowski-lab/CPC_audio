#!/usr/bin/env python3 -u

import logging
import os
from posixpath import join
import sys
import argparse
from itertools import chain
from pathlib import Path
import time
import copy
import numpy as np
import soundfile as sf

from cpc.feature_loader import loadModel, FeatureModule

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    parser.add_argument("path_checkpoint", type=str,
                        help="Path to the trained model checkpoint.")
    parser.add_argument("path_data", type=str,
                        help="Path to the dataset that we want to compute ABX for.")
    parser.add_argument("path_output_dir", type=str,
                        help="Path to the output directory.")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on a cpu machine.")
    parser.add_argument("--file_extension", type=str, default="wav",
                          help="Extension of the audio files in the dataset (default: wav).")
    parser.add_argument('--gru_level', type=int, default=-1,
                        help='Hidden level of the LSTM autoregressive model to be taken'
                        '(default: -1, last layer).')
    parser.add_argument('--nullspace', action='store_true',
                        help="Additionally load nullspace")
    return parser.parse_args()

def main():
    # Parse and print args
    args = parse_args()
    logger.info(args)

    # Load the model
    print("")
    print(f"Loading model from {args.path_checkpoint}") 

    if args.gru_level is not None and args.gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=args.gru_level)
    else:
        updateConfig = None

    model = loadModel([args.path_checkpoint], load_nullspace=args.nullspace, updateConfig=updateConfig)[0]
    
    if args.gru_level is not None and args.gru_level > 0:
        # Keep hidden units at LSTM layers on sequential batches
        if args.nullspace:
            model.cpc.gAR.keepHidden = True
        else:
            model.gAR.keepHidden = True
    
    model.disableSmartaveragingLossParameter()
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Register the hooks
    layer_outputs = {}
    def get_layer_output(name):
        def hook(model, input, output):
            if type(output) is tuple:
                layer_outputs[name] = output[0].detach().squeeze(1).cpu().numpy()
            elif type(output) is dict:
                layer_outputs[name] = output["x"].detach().squeeze(0).cpu().numpy()
            else:
                layer_outputs[name] = output.detach().squeeze(0).cpu().numpy()
        return hook

    layer_names = []
    layer_name = os.path.basename(os.path.dirname(args.path_checkpoint))
    layer_names.append(layer_name)
    if not args.nullspace:
        model.gAR.register_forward_hook(get_layer_output(layer_name))
    else:
        model.nullspace.register_forward_hook(get_layer_output(layer_name))

    model = model.eval().to(device)  
    print("Model loaded!")
    print(model)

    # Extract values from chosen layers and save them to files
    filenames = [os.path.join(root, file) for root, dirs, files in os.walk(args.path_data) for file in files if file.endswith(args.file_extension)]

    with torch.no_grad():     
        for i, f in enumerate(filenames):
            print("Progress {:2.1%}".format(i / len(filenames)), end="\r")
            input_f = os.path.join(args.path_data, f)
            x, sample_rate = sf.read(input_f)
            x = torch.tensor(x).float().reshape(1,1,-1).to(device)
            output = model(x, None)[0]

            for layer_name, value in layer_outputs.items():
                Path(args.path_output_dir).mkdir(parents=True, exist_ok=True)
                out_f = os.path.join(args.path_output_dir, os.path.splitext(os.path.basename(f))[0] + ".npy")
                np.save(out_f, value)

if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()
    main()

