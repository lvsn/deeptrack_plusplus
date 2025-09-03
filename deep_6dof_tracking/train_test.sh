#!/usr/bin/env bash

DATA_PATH=/gel/usr/chren50/dataset
OUTPUT_PATH=/home-local2/chren50.extra.nobkp/output_test/dragon258/trans_test1
BACKGROUND_PATH=/gel/usr/chren50/background
OCCLUDER_PATH=/gel/usr/chren50/hand


timestamp () {
    date +"%y-%m-%d_%Hh%Mm%Ss"
}

# conda activate tracking_test1

export PYTHONPATH=$PYTHONPATH:"/gel/usr/chren50/source/deep_6dof_tracking" 
echo $PYTHONPATH

LOG_FILE=${OUTPUT_PATH}/$(timestamp)_train.log
echo Writing logs in $LOG_FILE

train () {
    # cd tools
    # python3 dataset_mean.py -d $DATA_PATH/$1 -b $BACKGROUND_PATH -r $OCCLUDER_PATH
    # cd ..
    python3 train.py -e 25 \
            --output $OUTPUT_PATH \
            --dataset $DATA_PATH/$1 \
            --background $BACKGROUND_PATH \
            --occluder $OCCLUDER_PATH \
            --architecture res \
            --batchsize 128 \
            --tensorboard \
            -i 1 \
            --camera /gel/usr/chren50/source/deep_6dof_tracking/deep_6dof_tracking/data/sensors/camera_parameter_files/synthetic.json \
            --shader /gel/usr/chren50/source/deep_6dof_tracking/deep_6dof_tracking/data/shaders
}

echo "Start time: $(date)"
train dragon258
echo "End time: $(date)"