#!/usr/bin/env bash


SEQUENCE_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/laval6dof
MODEL_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/tracking_wacv/
BACKGROUND_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/SUN3D
OCCLUDER_PATH=/media/ssd/dataset/tracking/hand/train
network_name=generic_wacv_moddrop
net_model=res_moddrop

python3 latent_check.py  --architecture ${net_model} \
                         --backend cuda \
                         -o /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/to_delete \
                         --network ${MODEL_PATH}/${network_name}/model_last.pth.tar \
                         -m tools/model_configs/dragon.yml \
                         -b ${BACKGROUND_PATH} \
                         -s 1000 \
                         --show





