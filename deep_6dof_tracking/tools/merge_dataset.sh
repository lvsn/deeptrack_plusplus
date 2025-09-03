#!/usr/bin/env bash

DATA_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/deeptracking_eccv/dataset_storage

object=shoe

#python3 sequence_merge.py \
#     --dataset1 ${DATA_PATH}/${object}_top/train \
#     --dataset2 ${DATA_PATH}/${object}_bottom/train \
#     --output ${DATA_PATH}/${object}_real/train

#tar -xzf ${DATA_PATH}/${object}_t3r20.tar.gz -C ${DATA_PATH}
mkdir ${DATA_PATH}/${object}_real_synth
python3 sequence_merge.py \
     --dataset1 ${DATA_PATH}/${object}_real/train \
     --dataset2 ${DATA_PATH}/${object}_t3r20/train \
     --output ${DATA_PATH}/${object}_real_synth/train
#cp -r ${DATA_PATH}/${object}_t3r20/valid ${DATA_PATH}/${object}_real_synth/
#tar -czf ${DATA_PATH}/${object}_real_synth.tar.gz ${DATA_PATH}/${object}_real_synth
#rm -r ${DATA_PATH}/${object}_t3r20
#rm -r ${DATA_PATH}/${object}_real_synth