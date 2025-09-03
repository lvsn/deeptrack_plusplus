#!/usr/bin/env bash

OUTPUT_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Experiments/WACV_2019/Test_Choi
SEQUENCE_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/choi_christensen
MODEL_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Experiments/WACV_2019/Networks
GEOMETRY_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models

declare -a object_array=("kinect_box" "milk" "orange_juice" "tide")
declare -a width_array=(420 300 300 300)

declare -a multi_array=("generic_wacv_rgb" "generic_wacv_moddrop")

model_type=res_moddrop

cd ..

for model in "${multi_array[@]}"; do
    for ((i=0;i<${#object_array[@]};++i)); do
        object=${object_array[i]}
        width=${width_array[i]}
        mkdir -p ${OUTPUT_PATH}/${model}/${object}
        python3 sequence_test.py --video \
            -o ${OUTPUT_PATH}/${model}/${object} \
            -s ${SEQUENCE_PATH}/${object} \
            -m ${MODEL_PATH}/${model}/model_best.pth.tar \
            -g $GEOMETRY_PATH/${object} \
            --architecture ${model_type} --rgb --object_width ${width} # --show
    done
done