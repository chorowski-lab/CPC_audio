########## CHANGE THIS ##################
ZEROSPEECH_EVAL_ENVIRONMENT=zerospeech2021 # Where the zerospeech2021-evaluate is installed
CPC_ENVIRONMENT=202010-fairseq-c11
CONDA_PATH=/pio/scratch/2/i273233/miniconda3
#########################################

DATASET_PATH=false
ZEROSPEECH_DATASET_PATH=false
CHECKPOINT_PATH=false
SAVE_DIR=false
NULLSPACE=false
NO_TEST=false
FORCE_CPU=false

print_usage() {
  echo -e "Usage: ./eval_abx.sh"
  echo -e "\t-d DATASET_PATH (Either ZEROSPEECH_DATASET_PATH or LIBRISPEECH_FLATTENED_DATASET_PATH [Or anything that has directory structure of these two with dev-*.item files from ZEROSPEECH_DATASET_PATH])"
  echo -e "\t-r ZEROSPEECH_DATASET_PATH"
  echo -e "\t-c CHECKPOINT_PATH"
  echo -e "\t-o SAVE_DIR"
  echo -e "OPTIONAL ARGS:"
  echo -e "\t-n (Provide this flag if you want to load a model with nullspace)"
  echo -e "\t-a CONDA_PATH"
  echo -e "\t-e CPC_ENVIRONMENT"
  echo -e "\t-z ZEROSPEECH_EVAL_ENVIRONMENT (The conda environment where the zerospeech2021-evaluate is installed)"
  echo -e "\t-t (Do not compute embeddings for test set)"
}

while getopts 'd:r:c:o:na:e:z:tf' flag; do
    case "${flag}" in
        d) DATASET_PATH="${OPTARG}" ;;
        r) ZEROSPEECH_DATASET_PATH="${OPTARG}" ;;
        c) CHECKPOINT_PATH="${OPTARG}" ;;
        o) SAVE_DIR="${OPTARG}" ;;
        n) NULLSPACE=true ;;
        a) CONDA_PATH="${OPTARG}" ;;
        e) CPC_ENVIRONMENT="${OPTARG}" ;;
        z) ZEROSPEECH_EVAL_ENVIRONMENT="${OPTARG}" ;;
        t) NO_TEST=true ;;
        f) FORCE_CPU=true ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo $DATASET_PATH $ZEROSPEECH_DATASET_PATH $CHECKPOINT_PATH $SAVE_DIR $NULLSPACE $CONDA_PATH $CPC_ENVIRONMENT $ZEROSPEECH_EVAL_ENVIRONMENT $NO_TEST

if [[ $DATASET_PATH == false || $ZEROSPEECH_DATASET_PATH == false || $CHECKPOINT_PATH == false || $SAVE_DIR == false ]]
then
    echo "Either DATASET_PATH or ZEROSPEECH_DATASET_PATH or CHECKPOINT_PATH or SAVE_DIR is not set."
    print_usage
    exit 1
fi

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

results=$SAVE_DIR/results
embeddings=$SAVE_DIR/embeddings
mkdir -p embeddings

source $CONDA_PATH/etc/profile.d/conda.sh
SAVED_ENVIRONMENT=$(conda info | sed -n 's/\( \)*active environment : //p')
echo SAVED_ENVIRONMENT: $SAVED_ENVIRONMENT

ENV_TO_ACTIVATE=$CPC_ENVIRONMENT
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
python $SCRIPT_PATH/embeddings_abx.py $CHECKPOINT_PATH $DATASET_PATH $embeddings --gru_level 2 $params

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
            if [[ ! -f "$ZEROSPEECH_DATASET_PATH/phonetic/$directory/${filename_no_ext}.wav" ]] 
            then 
                rm $embeddings/$i/phonetic/$directory/$file 
            fi
        done
    done 
done

conda activate $ZEROSPEECH_EVAL_ENVIRONMENT

frame_shift="0.01"
echo "Frame shift is ${frame_shift}s"

metrics=("cosine" "euclidean")
#metrics=("cosine")
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
        #zerospeech2021-evaluate -j 12 -o $results/$metric/$i --force-cpu --no-lexical --no-syntactic --no-semantic $ZEROSPEECH_DATASET_PATH $embeddings/$i
        if [[ $FORCE_CPU == true ]]
        then
            zerospeech2021-evaluate -j 20 -o $results/$metric/$i --force-cpu --no-lexical --no-syntactic --no-semantic $ZEROSPEECH_DATASET_PATH $embeddings/$i
        else
            zerospeech2021-evaluate -j 20 -o $results/$metric/$i --no-lexical --no-syntactic --no-semantic $ZEROSPEECH_DATASET_PATH $embeddings/$i
        fi
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
done > $SAVE_DIR/combined_results.txt

conda activate $SAVED_ENVIRONMENT