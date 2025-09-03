#!/usr/bin/env bash

# 1:object name     2: network path     3: sequence path    4: output_path  5: reset frame 6:model_name
test () {
    mkdir -p $4
    python3 sequence_test.py --video -r $5 \
        -o $4 \
        -s $3 \
        -m $2/model_best.pth.tar \
        -g $GEOMETRY_PATH/$1 \
        --architecture $6 --rgb
}

test_reset () {
    mkdir -p $4
    python3 sequence_test.py --video -r $5 \
        -o $4 \
        -s $3 \
        -m $2/model_best.pth.tar \
        -g $GEOMETRY_PATH/$1 \
        --architecture $6 \
        --resetlost --rgb
}

OUTPUT_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Experiments/WACV_2019/Test
# dragon, turtle, walkman, wateringcan
SEQUENCE_PATH1=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/laval6dof
# clock, cookiejar, dog, kinect, lego, shoe, skull
SEQUENCE_PATH2=/media/mathieu/LaCie/Deeptrack/final/laval6dof/processed

MODEL_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Experiments/WACV_2019/Networks
GEOMETRY_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models

declare -a multi_array=("generic_wacv_moddrop" "generic_wacv_rgb")
declare -a sequences=(${SEQUENCE_PATH2})


net_model=res_moddrop

cd ..

for MODEL_NAME in "${multi_array[@]}"
do
    MODEL=${MODEL_PATH}/${MODEL_NAME}
        # iterate over all sequences
    for sequence_path in ${sequences}; do
        for sequence in ${sequence_path}/*; do
            sequence_name=${sequence##*/}
            result_path=${OUTPUT_PATH}/${MODEL_NAME}/${sequence_name}
            OBJECT_NAME=${sequence_name%%_*}
            # if output is already there, dont process (if figs is there, it means that everything was processed
            if [ -f ${result_path}/prediction_pose.csv ]
            then
                echo "Skip..."
            else
                echo "Process..."
                if [[ ${sequence_name} = *"occlusion"* ]]; then
                    #test ${OBJECT_NAME} ${MODEL} ${sequence} ${result_path} 15 ${net_model}
                    test_reset ${OBJECT_NAME} ${MODEL} ${sequence} ${result_path} 0 ${net_model}
                elif [[ ${sequence_name} = *"interaction"* ]]; then
                    test_reset ${OBJECT_NAME} ${MODEL} ${sequence} ${result_path} 0 ${net_model}
                else
                    test ${OBJECT_NAME} ${MODEL} ${sequence} ${result_path} 0 ${net_model}
                fi
            fi
        done
    done
done





