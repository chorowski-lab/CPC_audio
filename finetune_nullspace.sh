SPEAKERS="speakers_factorized"
PHONEMES="phonemes_nullspace"
SPEAKERS_NULLSPACE="speakers_nullspace"

DATASET_PATH=false
TRAIN_SPLIT_FILE_PATH=false
VALIDATION_SPLIT_FILE_PATH=false
BASELINE_NO_CLUSTERING_CHECKPOINT_PATH=false
SAVE_DIR=false
DIM_INBETWEEN=false
FROM_STEP=$SPEAKERS
PHONEME_ALIGNMENTS_FILE=false

print_usage() {
  echo -e "Usage: ./finetune_nullspace.sh"
  echo -e "\t-d DATASET_PATH (E.g. LIBRISPEECH_DATASET_PATH/train-clean-100)"
  echo -e "\t-t TRAIN_SPLIT_FILE_PATH (E.g. LIBRISPEECH_TRAIN_CLEAN_100_TRAIN_SPLIT_FILE_PATH)"
  echo -e "\t-v VALIDATION_SPLIT_FILE_PATH (E.g. LIBRISPEECH_TRAIN_CLEAN_100_TEST_SPLIT_FILE_PATH)"
  echo -e "\t-c BASELINE_NO_CLUSTERING_CHECKPOINT_PATH"
  echo -e "\t-o SAVE_DIR"
  echo -e "\t-n DIM_INBETWEEN (Dimension of nullspace will be DIM_EMBEDDING - DIM_INBETWEEN)"
  echo -e "\t-p PHONEME_ALIGNMENTS_FILE (Path to the file containing phonemes for the entire dataset)"
  echo -e "OPTIONAL ARGS:"
  echo -e "\t-f FROM_STEP (From which step do you want to start. Order: $SPEAKERS [default] -> $PHONEMES -> $SPEAKERS_NULLSPACE)"
}

while getopts 'd:t:v:c:o:n:f:p:' flag; do
    case "${flag}" in
        d) DATASET_PATH="${OPTARG}" ;;
        t) TRAIN_SPLIT_FILE_PATH="${OPTARG}" ;;
        v) VALIDATION_SPLIT_FILE_PATH="${OPTARG}" ;;
        c) BASELINE_NO_CLUSTERING_CHECKPOINT_PATH="${OPTARG}" ;;
        o) SAVE_DIR="${OPTARG}" ;;
        n) DIM_INBETWEEN="${OPTARG}" ;;
        f) FROM_STEP="${OPTARG}" ;;
        p) PHONEME_ALIGNMENTS_FILE="${OPTARG}" ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo $DATASET_PATH $TRAIN_SPLIT_FILE_PATH $VALIDATION_SPLIT_FILE_PATH $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH $SAVE_DIR $DIM_INBETWEEN $FROM_STEP $PHONEME_ALIGNMENTS_FILE

if [[ $DATASET_PATH == false || $TRAIN_SPLIT_FILE_PATH == false || $VALIDATION_SPLIT_FILE_PATH == false || $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH == false || $SAVE_DIR == false || $DIM_INBETWEEN == false || $PHONEME_ALIGNMENTS_FILE == false ]]
then
    echo "Either DATASET_PATH, TRAIN_SPLIT_FILE_PATH, VALIDATION_SPLIT_FILE_PATH, BASELINE_NO_CLUSTERING_CHECKPOINT_PATH, SAVE_DIR, DIM_INBETWEEN or PHONEME_ALIGNMENTS_FILE is not set."
    print_usage
    exit 1
fi

mkdir -p $SAVE_DIR

case $FROM_STEP in
$SPEAKERS)
    echo $SPEAKERS
    mkdir -p ${SAVE_DIR}/${SPEAKERS}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SPLIT_FILE_PATH $VALIDATION_SPLIT_FILE_PATH $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH --pathCheckpoint ${SAVE_DIR}/${SPEAKERS}_${DIM_INBETWEEN} --mode $SPEAKERS --max_size_loaded 40000000 --n_process_loader 2 --model cpc --dim_inter $DIM_INBETWEEN --gru_level 2
    ;&
$PHONEMES)
    echo $PHONEMES
    mkdir -p ${SAVE_DIR}/${PHONEMES}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SPLIT_FILE_PATH $VALIDATION_SPLIT_FILE_PATH $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH --pathCheckpoint ${SAVE_DIR}/${PHONEMES}_${DIM_INBETWEEN} --mode $PHONEMES --max_size_loaded 40000000 --n_process_loader 2 --model cpc --pathPhone $PHONEME_ALIGNMENTS_FILE --path_speakers_factorized ${SAVE_DIR}/${SPEAKERS}_${DIM_INBETWEEN}/checkpoint_9.pt --dim_inter $DIM_INBETWEEN --gru_level 2
    ;&
$SPEAKERS_NULLSPACE)
    echo $SPEAKERS_NULLSPACE
    mkdir -p ${SAVE_DIR}/${SPEAKERS_NULLSPACE}_${DIM_INBETWEEN}
    python cpc/eval/linear_separability.py $DATASET_PATH $TRAIN_SPLIT_FILE_PATH $VALIDATION_SPLIT_FILE_PATH $BASELINE_NO_CLUSTERING_CHECKPOINT_PATH --pathCheckpoint ${SAVE_DIR}/${SPEAKERS_NULLSPACE}_${DIM_INBETWEEN} --mode $SPEAKERS_NULLSPACE --max_size_loaded 40000000 --n_process_loader 2 --model cpc --path_speakers_factorized ${SAVE_DIR}/${SPEAKERS}_${DIM_INBETWEEN}/checkpoint_9.pt --dim_inter $DIM_INBETWEEN --gru_level 2
    ;;
*)
    echo "Invalid from step: ${FROM_STEP} while it should be either ${SPEAKERS}, ${PHONEMES} or ${SPEAKERS_NULLSPACE}"
    ;;
esac

echo "Checkpoint with nullspace is located in ${SAVE_DIR}/${PHONEMES}_${DIM_INBETWEEN}/checkpoint_9.pt"
echo "The results of all the experiments are located in ${SAVE_DIR}/DIRECTORY/checkpoint_logs.json"

exit 0