NULLSPACE_SIZE=$1
BATCH_SIZE_GPU=$2
MAX_ITER=$3

python cpc/criterion/clustering/clustering_script.py     --pathDB $zd/LibriSpeech/train-clean-100/ --recursionLevel 1     --nClusters 50 --MAX_ITER $MAX_ITER --level_gru 2     --save --load --batchSizeGPU $BATCH_SIZE_GPU --max_size_loaded 40000000 --n_process_loader 2 --nullspace     ../linear_separability/cpc/gru_level2/cpc_official_phonemes_nullspace_$NULLSPACE_SIZE/checkpoint_9.pt     checkpoints/clustering_CPC_big_kmeans50_nullspace_$NULLSPACE_SIZE/clustering_CPC_big_kmeans50_nullspace_$NULLSPACE_SIZE.pt
for directory in dev-clean dev-other test-clean test-other train-clean-100 train-full-960
do
	python ./scripts/quantize_audio.py $cpc/checkpoints/clustering_CPC_big_kmeans50_nullspace_$NULLSPACE_SIZE/clustering_CPC_big_kmeans50_nullspace_$NULLSPACE_SIZE.pt $zd/LibriSpeech/$directory/ /pio/gluster/i273233/quantized/nullspace_$NULLSPACE_SIZE/LibriSpeech/$directory --file_extension flac --nobatch --nullspace
done