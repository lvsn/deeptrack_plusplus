#!/usr/bin/env bash

SPECIFIC_TAG=specific_specular
OUTPUT_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/tracking_choi
SEQUENCE_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/choi_christensen
MODEL_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/models/tracking
GEOMETRY_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models

declare -a object_array=("milk")

model=generic_moddropv2_2_nonorm
model_type=res_moddrop2

for object in "${object_array[@]}"
do
    #object=kinect_box

    # kinect box width : 300

    mkdir -p ${OUTPUT_PATH}/${object}
    python3 sequence_test.py --video \
        -o ${OUTPUT_PATH}/${object} \
        -s ${SEQUENCE_PATH}/${object} \
        -m ${MODEL_PATH}/${model}/model_best.pth.tar \
        -g $GEOMETRY_PATH/${object} \
        --architecture ${model_type} --rgb --show #--object_width 300
done