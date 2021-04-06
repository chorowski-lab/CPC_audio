import os
import sys
import shutil
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("librispeech_path", type=str,
                        help="Path to the root directory of LibriSpeech.")
    parser.add_argument("zerospeech_dataset_path", type=str,
                        help="Path to the ZeroSpeech dataset.")
    parser.add_argument("target_path", type=str,
                        help="Path to the output directory.")
    parser.add_argument("--file_extension", type=str, default="flac",
                          help="Extension of the audio files in the dataset (default: flac).")
    return parser.parse_args()

def main():
    # Parse and print args
    args = parse_args()
    #logger.info(args)

    phonetic = "phonetic"
    datasets = ["dev-clean", "dev-other", "test-clean", "test-other"]

    for dataset in datasets:
        print("> {}".format(dataset))
        target_dirname = os.path.join(args.target_path, phonetic, dataset)
        Path(target_dirname).mkdir(parents=True, exist_ok=True)

        librispeech_dirname = os.path.join(args.librispeech_path, dataset)
        files = [(filename, dirname) for dirname, _, files in os.walk(librispeech_dirname, followlinks=True) for filename in files if filename.endswith(args.file_extension)]
        for i, (filename, dirname) in enumerate(files):
            print("Progress {:2.1%}".format(i / len(files)), end="\r")
            input_path = os.path.join(dirname, filename)
            output_path = os.path.join(target_dirname, os.path.splitext(filename)[0] + ".wav")
            data, sample_rate = sf.read(input_path)
            sf.write(output_path, data, sample_rate)

        if dataset.startswith("dev"):
            source_item_path = os.path.join(args.zerospeech_dataset_path, phonetic, dataset, dataset + ".item")
            target_item_path = os.path.join(target_dirname, dataset + ".item")
            shutil.copy(source_item_path, target_item_path)


if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()
    main()