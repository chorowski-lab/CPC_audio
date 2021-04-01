SPEAKERS="speakers_factorized"
PHONEMES="phonemes_nullspace"
SPEAKERS_NULLSPACE="speakers_nullspace"

DATASET_PATH=false
TRAIN_SET=false
VALIDATION_SET=false
CHECKPOINT_PATH=false
OUTPUT_DIR=false
DIM_INBETWEEN=false
FROM_STEP=$SPEAKERS
PHONES_PATH=false

print_usage() {
  echo -e "Usage: ./finetune_nullspace.sh"
  echo -e "\t-d DATASET_PATH"
  echo -e "\t-t TRAIN_SET"
  echo -e "\t-v VALIDATION_SET"
  echo -e "\t-c CHECKPOINT_PATH"
  echo -e "\t-o OUTPUT_DIR"
  echo -e "\t-n DIM_INBETWEEN (Dimension of nullspace will be DIM_EMBEDDING - DIM_INBETWEEN)"
  echo -e "OPTIONAL FLAGS:"
  echo -e "\t-f FROM_STEP (From which step do you want to start. Order: $SPEAKERS -> $PHONEMES -> $SPEAKERS_NULLSPACE)"
  echo -e "\t-p PHONES_PATH (Path to the file containing phonemes for the entire dataset. You don't need it if you start from $SPEAKERS_NULLSPACE)"
}

while getopts 'd:t:v:c:o:n:f:p:' flag; do
    case "${flag}" in
        d) DATASET_PATH="${OPTARG}" ;;
        t) TRAIN_SET="${OPTARG}" ;;
        v) VALIDATION_SET="${OPTARG}" ;;
        c) CHECKPOINT_PATH="${OPTARG}" ;;
        o) OUTPUT_DIR="${OPTARG}" ;;
        n) DIM_INBETWEEN="${OPTARG}" ;;
        f) FROM_STEP="${OPTARG}" ;;
        p) PHONES_PATH="${OPTARG}" ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo $DATASET_PATH $TRAIN_SET $VALIDATION_SET $CHECKPOINT_PATH $OUTPUT_DIR $DIM_INBETWEEN $FROM_STEP $PHONES_PATH

if [[ $DATASET_PATH == false || $TRAIN_SET == false || $VALIDATION_SET == false || $CHECKPOINT_PATH == false || $OUTPUT_DIR == false  || $DIM_INBETWEEN == false || ( $PHONES_PATH == false && $FROM_STEP != $SPEAKERS ) ]]
then
    echo "Either DATASET_PATH, TRAIN_SET, VALIDATION_SET, CHECKPOINT_PATH, OUTPUT_DIR or DIM_INBETWEEN is not set or there are invalid PHONES_PATH and FROM_STEP."
    print_usage
    exit 1
fi

mkdir -p $OUTPUT_DIR

case $FROM_STEP in
$SPEAKERS)
    echo $SPEAKERS
    mkdir -p ${OUTPUT_DIR}/${SPEAKERS}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SET $VALIDATION_SET $CHECKPOINT_PATH --pathCheckpoint ${OUTPUT_DIR}/${SPEAKERS}_${DIM_INBETWEEN} --mode $SPEAKERS --max_size_loaded 40000000 --n_process_loader 2 --model cpc --dim_inter $DIM_INBETWEEN --gru_level 2
    ;&
$PHONEMES)
    echo $PHONEMES
    mkdir -p ${OUTPUT_DIR}/${PHONEMES}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SET $VALIDATION_SET $CHECKPOINT_PATH --pathCheckpoint ${OUTPUT_DIR}/${PHONEMES}_${DIM_INBETWEEN} --mode $PHONEMES --max_size_loaded 40000000 --n_process_loader 2 --model cpc --pathPhone $PHONES_PATH --path_speakers_factorized ${OUTPUT_DIR}/${SPEAKERS}_${DIM_INBETWEEN}/checkpoint_9.pt --dim_inter $DIM_INBETWEEN --gru_level 2
    ;&
$SPEAKERS_NULLSPACE)
    echo $SPEAKERS_NULLSPACE
    mkdir -p ${OUTPUT_DIR}/${SPEAKERS_NULLSPACE}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SET $VALIDATION_SET $CHECKPOINT_PATH --pathCheckpoint ${OUTPUT_DIR}/${SPEAKERS_NULLSPACE}_${DIM_INBETWEEN} --mode $SPEAKERS_NULLSPACE --max_size_loaded 40000000 --n_process_loader 2 --model cpc --path_speakers_factorized ${OUTPUT_DIR}/${SPEAKERS}_${DIM_INBETWEEN}/checkpoint_9.pt --dim_inter $DIM_INBETWEEN --gru_level 2
    ;;
*)
    echo "Invalid from step: ${FROM_STEP} while it should be either ${SPEAKERS}, ${PHONEMES} or ${SPEAKERS_NULLSPACE}"
    ;;
esac

echo "Checkpoint with nullspace is located in ${OUTPUT_DIR}/${PHONEMES}_${DIM_INBETWEEN}/checkpoint_9.pt"
echo "The results of all the experiments are located in ${OUTPUT_DIR}/DIRECTORY/checkpoint_logs.json"

exit 0