
for deg in 0 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo $deg
    mkdir ${centerpushDir}/phoneme_classif_nonull_${deg}/
    python cpc/eval/linear_separability.py $zd/LibriSpeech/train-clean-100/ \
    $zd/LibriSpeech/labels_split/train_split_100.txt \
    $zd/LibriSpeech/labels_split/test_split_100.txt \
    $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
    --centerpushFile $zd/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
    --centerpushDeg $deg \
    --pathCheckpoint ${centerpushDir}/phoneme_classif_nonull_${deg}/ \
    --mode phonemes --max_size_loaded 40000000 --n_process_loader 2 \
    --model cpc --pathPhone $zd/LibriSpeech/alignments2/converted_aligned_phones.txt \
    --gru_level 2 --batchSizeGPU 32 | tee ${centerpushDir}/phoneme_classif_nonull_${deg}/log.txt
done