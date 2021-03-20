SAVE_DIR="/pio/scratch/1/i273233/linear_separability/cpc/gru_level2/cpc_official"
SPEAKERS="speakers_factorized"
PHONEMES="phonemes_nullspace"
SPEAKERS_NULLSPACE="speakers_nullspace"

DIM_INTER=$1
FROM_STEP=$SPEAKERS
if [[ $# -ge 2 ]]; then
    FROM_STEP=$2
fi

case $FROM_STEP in
$SPEAKERS)
    echo $SPEAKERS
    mkdir -p ${SAVE_DIR}_${SPEAKERS}_${DIM_INTER} && python cpc/eval/linear_separability.py $zd/LibriSpeech/train-clean-100/ $zd/LibriSpeech/labels_split/train_split_100.txt $zd/LibriSpeech/labels_split/test_split_100.txt $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt --pathCheckpoint ${SAVE_DIR}_${SPEAKERS}_${DIM_INTER} --mode $SPEAKERS --max_size_loaded 40000000 --n_process_loader 2 --model cpc --dim_inter $DIM_INTER --gru_level 2 | tee ${SAVE_DIR}_${SPEAKERS}_${DIM_INTER}/log.txt
    ;&
$PHONEMES)
    echo $PHONEMES
    mkdir -p ${SAVE_DIR}_${PHONEMES}_${DIM_INTER} && python cpc/eval/linear_separability.py $zd/LibriSpeech/train-clean-100/ $zd/LibriSpeech/labels_split/train_split_100.txt $zd/LibriSpeech/labels_split/test_split_100.txt $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt --pathCheckpoint ${SAVE_DIR}_${PHONEMES}_${DIM_INTER} --mode $PHONEMES --max_size_loaded 40000000 --n_process_loader 2 --model cpc --pathPhone $zd/LibriSpeech/alignments2/converted_aligned_phones.txt --path_speakers_factorized ${SAVE_DIR}_${SPEAKERS}_${DIM_INTER}/checkpoint_9.pt --dim_inter $DIM_INTER --gru_level 2 | tee ${SAVE_DIR}_${PHONEMES}_${DIM_INTER}/log.txt
    ;&
$SPEAKERS_NULLSPACE)
    echo $SPEAKERS_NULLSPACE
    mkdir -p ${SAVE_DIR}_${SPEAKERS_NULLSPACE}_${DIM_INTER} && python cpc/eval/linear_separability.py $zd/LibriSpeech/train-clean-100/ $zd/LibriSpeech/labels_split/train_split_100.txt $zd/LibriSpeech/labels_split/test_split_100.txt $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt --pathCheckpoint ${SAVE_DIR}_${SPEAKERS_NULLSPACE}_${DIM_INTER} --mode $SPEAKERS_NULLSPACE --max_size_loaded 40000000 --n_process_loader 2 --model cpc --path_speakers_factorized ${SAVE_DIR}_${SPEAKERS}_${DIM_INTER}/checkpoint_9.pt --dim_inter $DIM_INTER --gru_level 2 | tee ${SAVE_DIR}_${SPEAKERS_NULLSPACE}_${DIM_INTER}/log.txt
    ;;
*)
    echo "Invalid from step: ${FROM_STEP} while it should be either ${SPEAKERS}, ${PHONEMES} or ${SPEAKERS_NULLSPACE}"
    ;;
esac

exit 0