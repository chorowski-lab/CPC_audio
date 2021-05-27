



python train.py --pathDB /pio/scratch/1/i283340/MGR/zs/ds2 \
--pathTrain /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
--pathVal /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
--file_extension .flac \
--normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
--max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
--schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
--pathCheckpoint /pio/gluster/i283340/cpcfcmtries/spam003/ \
--nEpoch 7 --logging_step 5 \
--FCMsettings --FCMhierARshorten 3. \
--FCMsegmentCostModule --FCMsegment_batchesMem 13 \
--supervised_classif_metric \
--path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
--linsepBatchSizeGPU 32 --linsep_n_epoch 1 --linsep_times 2 \
--linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/logs2-001 \
--linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/checkp2-001 \
--linsep_classif_each_epochs 8 \
--overrideArgsFile #--FCMhierARgradualStart 1


# python train.py --pathDB /pio/scratch/1/i283340/MGR/zs/ds2 \
# --pathTrain /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
# --pathVal /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
# --file_extension .flac \
# --normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
# --max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
# --schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
# --pathCheckpoint /pio/gluster/i283340/cpcfcmtries/spam003/ \
# --nEpoch 6 --logging_step 5 \
# --FCMsettings --FCMhierARshorten 3. \
# --supervised_classif_metric \
# --path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
# --linsepBatchSizeGPU 32 --linsep_n_epoch 1 --linsep_times 2 \
# --linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/logs2-001 \
# --linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/checkp2-001 \
# --linsep_classif_each_epochs 7 \
# --overrideArgsFile #--FCMhierARgradualStart 1


# python train.py --pathDB /pio/scratch/1/i283340/MGR/zs/ds2 \
# --pathTrain /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
# --pathVal /pio/scratch/1/i283340/MGR/zs/sometries/ds2part.txt \
# --file_extension .flac \
# --normMode layerNorm --dropout --rnnMode transformer --n_process_loader 1 \
# --max_size_loaded 4000000000 --nLevelsGRU 2 --batchSizeGPU 32 --limitNegsInBatch 8 \
# --schedulerRamp 10 --nPredicts 12 --CPCCTC --CPCCTCNumMatched 12 \
# --pathCheckpoint /pio/gluster/i283340/cpcfcmtries/spam003/ \
# --nEpoch 6 \
# --supervised_classif_metric \
# --path_phone_data /pio/scratch/1/i283340/MGR/zs/phones/converted_aligned_phones.txt \
# --linsepBatchSizeGPU 32 --linsep_n_epoch 1 --linsep_times 2 \
# --linsep_logs_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/logs2-001 \
# --linsep_checkpoint_dir /pio/gluster/i283340/cpcfcmtries/spam003/linsep/checkp2-001 \
# --linsep_classif_each_epochs 7 \
# --overrideArgsFile