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

SPECIFIC_TAG=specific_specular
OUTPUT_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/tracking_wacv
SEQUENCE_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/laval6dof
#SEQUENCE_PATH=/media/mathieu/LaCie/Deeptrack/final/laval6dof/to_evaluate
MODEL_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/models/tracking
GEOMETRY_PATH=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models

declare -a object_array=("dragon")
suffix="_specular_split1_flip"
#declare -a multi_array=("generic_rgb_iccv" "generic_moddrop_rgb_iccv")
#declare -a multi_array=("generic_da4_best_iccv")
declare -a multi_array=("generic_moddropv2_nonorm")

net_model=res_moddrop

#iterate over all objects
#for OBJECT_NAME in "${object_array[@]}"
#do
#    model=${MODEL_PATH}/${OBJECT_NAME}${suffix}
#
#    # iterate over all sequences
#    for sequence in ${SEQUENCE_PATH}/${OBJECT_NAME}*; do
#        sequence_name=${sequence##*${OBJECT_NAME}_}
#        obj_sequence_name=${sequence##*/}
#        echo Object: $OBJECT_NAME Test: $obj_sequence_name Model: ${net_model}
#        result_path=${OUTPUT_PATH}/${SPECIFIC_TAG}/${obj_sequence_name}
#        # if output is already there, dont process (if figs is there, it means that everything was processed
#        if [ -f ${result_path}/prediction_pose.csv ]
#        then
#            echo "Skip..."
#        else
#            if [[ $sequence_name = *"occlusion"* ]]; then
#                test ${OBJECT_NAME} $model $sequence $result_path 15 ${net_model}
#            elif [[ $sequence_name = *"hard"* ]]; then
#                test_reset ${OBJECT_NAME} $model $sequence $result_path 0 ${net_model}
#            elif [[ $sequence_name = *"interaction"* ]]; then
#                test ${OBJECT_NAME} $model $sequence $result_path 15 ${net_model}
#            else
#                test ${OBJECT_NAME} $model $sequence $result_path 0 ${net_model}
#            fi
#        fi
#    done
#done

for MODEL_NAME in "${multi_array[@]}"
do
    MODEL=${MODEL_PATH}/${MODEL_NAME}
        # iterate over all sequences
    for sequence in ${SEQUENCE_PATH}/*; do
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





