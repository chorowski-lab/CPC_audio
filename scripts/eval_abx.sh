########## CHANGE THIS ##################
ZEROSPEECH_EVAL_ENV=zerospeech2021 # Where the zerospeech2021-evaluate is installed
CPC_ENV=202010-fairseq-c11
CONDA_PATH=/pio/scratch/2/i273233/miniconda3
#########################################

DATASET_PATH=false
ORIGINAL_DATASET_PATH=false
CHECKPOINT_PATH=false
OUTPUT_DIR=false
NULLSPACE=false
NO_TEST=false
AUDIO_FORMAT=flac

print_usage() {
  echo -e "Usage: ./eval_abx.sh"
  echo -e "\t-d DATASET_PATH (Either ZEROSPEECH_DATASET_PATH or LIBRISPEECH_FLATTENED_DATASET_PATH [Or anything that has directory structure of these two with dev-*.item files from ZEROSPEECH_DATASET_PATH])"
  echo -e "\t-r ORIGINAL_DATASET_PATH"
  echo -e "\t-c CHECKPOINT_PATH"
  echo -e "\t-o OUTPUT_DIR"
  echo -e "OPTIONAL FLAGS:"
  echo -e "\t-n (Provide this flag if you want to load a model with nullspace)"
  echo -e "\t-a CONDA_PATH"
  echo -e "\t-e CPC_ENV"
  echo -e "\t-z ZEROSPEECH_EVAL_ENV (The conda environment where the zerospeech2021-evaluate is installed)"
  echo -e "\t-t (Do not compute embeddings for test set)"
  echo -e "\t-f audio files format in LibriSpeech dataset (without a dot)"
}

while getopts 'd:r:c:o:na:e:z:tf:' flag; do
    case "${flag}" in
        d) DATASET_PATH="${OPTARG}" ;;
        r) ORIGINAL_DATASET_PATH="${OPTARG}" ;;
        c) CHECKPOINT_PATH="${OPTARG}" ;;
        o) OUTPUT_DIR="${OPTARG}" ;;
        n) NULLSPACE=true ;;
        a) CONDA_PATH="${OPTARG}" ;;
        e) CPC_ENV="${OPTARG}" ;;
        z) ZEROSPEECH_EVAL_ENV="${OPTARG}" ;;
        t) NO_TEST=true ;;
        f) AUDIO_FORMAT=${OPTARG} ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo $DATASET_PATH $ORIGINAL_DATASET_PATH $CHECKPOINT_PATH $OUTPUT_DIR $NULLSPACE $CONDA_PATH $CPC_ENV $ZEROSPEECH_EVAL_ENV $NO_TEST

if [[ $DATASET_PATH == false || $ORIGINAL_DATASET_PATH == false || $CHECKPOINT_PATH == false || $OUTPUT_DIR == false ]]
then
    echo "Either DATASET_PATH or ORIGINAL_DATASET_PATH or CHECKPOINT_PATH or OUTPUT_DIR is not set."
    print_usage
    exit 1
fi

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

results=$OUTPUT_DIR/results
embeddings=$OUTPUT_DIR/embeddings
mkdir -p embeddings

source $CONDA_PATH/etc/profile.d/conda.sh
SAVED_ENV=$(conda info | sed -n 's/\( \)*active environment : //p')
echo SAVED_ENV: $SAVED_ENV

ENV_TO_ACTIVATE=$CPC_ENV
conda activate $ENV_TO_ACTIVATE

params=""
if [[ $NULLSPACE == true ]]
then
    params="${params} --nullspace"
fi

if [[ $NO_TEST == true ]]
then
    params="${params} --no_test"
fi
echo "Params: $params"

echo "$SCRIPT_PATH/embeddings_abx.py"
python $SCRIPT_PATH/embeddings_abx.py $CHECKPOINT_PATH $DATASET_PATH $embeddings --gru_level 2 --file_extension $AUDIO_FORMAT $params

directories=("dev-clean" "dev-other")
if [[ $NO_TEST == false ]]
then
    directories+=("test-clean" "test-other")
fi
echo "Directories: ${directories[@]}"

for i in `basename -a $(ls -d $embeddings/*/)`
do
    for directory in ${directories[@]}
    do 
        for file in `ls $embeddings/$i/phonetic/$directory` 
        do 
            filename_no_ext="${file%.*}" 
            if [[ ! -f "$ORIGINAL_DATASET_PATH/phonetic/$directory/${filename_no_ext}.$AUDIO_FORMAT" ]] 
            then 
                rm $embeddings/$i/phonetic/$directory/$file 
            fi
        done
    done 
done

conda activate $ZEROSPEECH_EVAL_ENV

frame_shift="0.01"
echo "Frame shift is ${frame_shift}s"

metrics=("cosine" "euclidean")
for metric in ${metrics[@]}
do
    cat > $embeddings/$metric.yaml << EOF
author: LSTM Baseline
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  LSTM. See https://zerospeech.com/2021 for more details.
open_source: true
train_set: librispeech 100 and 960
gpu_budget: 60
parameters:
  phonetic:
    metric: ${metric}
    frame_shift: ${frame_shift}
EOF

    for i in `basename -a $(ls -d $embeddings/*/)`
    do
        cp $embeddings/$metric.yaml $embeddings/$i/meta.yaml
        #zerospeech2021-evaluate -j 12 -o $results/$metric/$i --no-lexical --no-syntactic --no-semantic $DATASET_PATH $embeddings/$i
        #zerospeech2021-evaluate -j 12 -o $results/$metric/$i --force-cpu --no-lexical --no-syntactic --no-semantic $ORIGINAL_DATASET_PATH $embeddings/$i
        #zerospeech2021-evaluate -j 20 -o $results/$metric/$i --force-cpu --no-lexical --no-syntactic --no-semantic $ORIGINAL_DATASET_PATH $embeddings/$i
        zerospeech2021-evaluate --force-cpu -j 20 -o $results/$metric/$i --no-lexical --no-syntactic --no-semantic $ORIGINAL_DATASET_PATH $embeddings/$i
    done
done

for metric in ${metrics[@]}
do
    for i in `basename -a $(ls -d $embeddings/*/)`
    do 
        echo $i $metric
        cat $results/$metric/$i/score_phonetic.csv
        echo
    done
done > $OUTPUT_DIR/combined_results.txt

conda activate $SAVED_ENV