

# RUN ON 1 GPU TO AVOID PRINTOUT INTERLACES

RUN_NAME="pushbothsq0-001"

python train.py --pathDB /pio/data/zerospeech2021/LibriSpeech/train-clean-100 \
--pathTrain /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/train_split.txt \
--pathVal /pio/scratch/2/jch/wav2vec/LibriSpeech100_labels_split/test_split.txt \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--supervised_classif_metric \
--speaker_sep --path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
--linsepBatchSizeGPU 32 --linsep_n_epoch 10 \
--linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/linsep/logs \
--linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/linsep/checkp \
--linsep_classif_each_epochs 20 \
--pathCheckpoint /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}/checkp/ \
--nEpoch 81 \
--FCMproject \
--FCMprotos 50 --FCMpushLossWeightEnc 0.001 --FCMpushLossWeightCtx 0.001 \
2>&1 | tee -ai /pio/gluster/i283340/cpcfcmtries/pushloss2/${RUN_NAME}.log
# --FCMprotos 50 --FCMpushLossLinear --FCMpushLossWeightEnc 0.0001 --FCMpushLossWeightCtx 0.0001
#--FCMmAfterAR 2. --FCMpushDegAllAfterAR 0.3 --FCMprotos 48 --FCMreprsConcat --FCMreprsConcatNormSumsNotLengths

# mBefore without concat - test with more protos? to make lstm bigger (or add some option for bigger LSTM and projection?)
#                          or just try cutting protos on that (with bigger num of protos)!
# some configs (change last line above):
#      // set fcmDebug in the CPCmodel to true!
# --FCMpushDegFeatureBeforeAR 0.3 --FCMpushDegCtxAfterAR 0.3 --FCMprotos 48
# --FCMmBeforeAR 2. (--FCMpushDegFeatureBeforeAR 0.3) --FCMprotos 256 --FCMleaveProtos 48
# --FCMmAfterAR 2. (--FCMpushDegFeatureBeforeAR 0.3 --FCMpushDegCtxAfterAR 0.3) --FCMprotos 48
# --FCMmAfterAR 2. (--FCMpushDegAllAfterAR 0.3) --FCMprotos 48
# [!] in all below additionally can add --FCMreprsConcatNormSumsNotLengths
# --FCMmBeforeAR 2. (--FCMpushDegFeatureBeforeAR 0.3 --FCMpushDegCtxAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat --hiddenEncoder 208
# --FCMmBeforeAR 2. (--FCMpushDegAllAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat --hiddenEncoder 208
# --FCMmAfterAR 2. (--FCMpushDegFeatureBeforeAR 0.3 --FCMpushDegCtxAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat  # here could add option for dim decrease only after AR
# --FCMmAfterAR 2. (--FCMpushDegAllAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat
# --FCMmAfterAR 2. (--FCMpushDegFeatureBeforeAR 0.3 --FCMpushDegCtxAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat --hiddenEncoder 208
# --FCMmAfterAR 2. (--FCMpushDegAllAfterAR 0.3) --FCMprotos 48 --FCMreprsConcat --hiddenEncoder 208


# --supervised_classif_metric \
# --speaker_sep --path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
# --linsepBatchSizeGPU 32 --linsep_n_epoch 10 \
# --linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/spam001-001linsep/logs/ \
# --linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/spam001-001linsep/checkp/ \
# --linsep_classif_each_epochs 25

# [!] for transformer in criterion has to be k*8 I guess for some reason
# 201 epochs to make it classify after 201th as it has number 200 ("each_epochs" checks %)

# max size loaded is not small enough if loading more things it seems

# $ head /pio/scratch/2/jch/wav2vec/runs/cpc_base/ls100_cpcctc_match12_pred8/out.txt 
# train_ls100.sh --normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 --max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 --schedulerRamp 10 --nPredicts 8 --CPCCTC --CPCCTCNumMatched 12 --pathCheckpoint /pio/scratch/2/jch/wav2vec/runs/cpc_base/ls100_cpcctc_match12_pred8
# /home/jch/scratch/jsalt/anaconda3/envs/202010-fairseq-c11/lib/python3.7/site-packages/torchaudio/backend/utils.py:54: UserWarning: "sox" backend is being deprecated. The default backend will be changed to "sox_io" backend in 0.8.0 and "sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. Please refer to https://github.com/pytorch/audio/issues/903 for the detail.
#   '"sox" backend is being deprecated. '
# Let's use 2 GPUs!
# No checkpoints found at /pio/scratch/2/jch/wav2vec/runs/cpc_base/ls100_cpcctc_match12_pred8
# CONFIG:
# {
#     "CPCCTC": true,
#     "CPCCTCNumMatched": 12,
#     "CPCCTCSelfLoop": false,
