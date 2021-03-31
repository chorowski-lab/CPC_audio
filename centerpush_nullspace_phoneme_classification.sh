
for deg in 0 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo $deg
    mkdir ${centerpushDir}/phoneme_classif_null_${deg}/
    python cpc/eval/linear_separability.py $zd/LibriSpeech/train-clean-100/ \
    $zd/LibriSpeech/labels_split/train_split_100.txt \
    $zd/LibriSpeech/labels_split/test_split_100.txt \
    $zd/checkpoints/CPC-big-kmeans50/cpc_ll6k/checkpoint_32.pt \
    --centerpushFile $cpcClustDir/checkpoints/clustering_CPC_big_kmeans50_nullspace_64/clustering_CPC_big_kmeans50_nullspace_64.pt \
    --centerpushDeg $deg \
    --pathCheckpoint ${centerpushDir}/phoneme_classif_null_${deg}/ \
    --mode phonemes_nullspace --max_size_loaded 40000000 --n_process_loader 2 \
    --model cpc --pathPhone $zd/LibriSpeech/alignments2/converted_aligned_phones.txt \
    --path_speakers_factorized $nullspaceDir/linear_separability/cpc/gru_level2/cpc_official_speakers_factorized_64/checkpoint_9.pt \
    --dim_inter 64 --gru_level 2 --batchSizeGPU 32 | tee ${centerpushDir}/phoneme_classif_null_${deg}/log.txt
done

