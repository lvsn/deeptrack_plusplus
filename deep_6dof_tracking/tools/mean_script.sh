#!/usr/bin/env bash
CODE_PATH="`( pwd )`"
SAVE_PATH=/home-local2/chren50.extra.nobkp/dataset

export PYTHONPATH=$PYTHONPATH:"/gel/usr/chren50/source/deep_6dof_tracking" 

generate () {
    echo "Running $4"
    mkdir $DATA_PATH
    python3 dataset_generator.py -o $1/train -m model_configs/$5 -s 200000 \
        -t $2 -r $3 --boundingbox $6 -e $7 --saveformat $8 --specular --minradius 0.4
    python3 dataset_generator.py -o $1/valid -m model_configs/$5 -s 20000 \
        -t $2 -r $3 --boundingbox $6 -e $7 --saveformat $8 --specular --minradius 0.4
    #cd $SAVE_PATH
    #tar -czf $4.tar.gz $4
    #mv $4.tar.gz $STORING_PATH
    #rm -r $4/
    #cd $CODE_PATHd
    #echo Stop $4 $(timestamp) >> $LOG_FILE
}


MODEL_FILE=clock.yml
OBJECT=clock

NAME=${OBJECT}
DATA_PATH=$SAVE_PATH/${NAME}
generate $DATA_PATH 0.02 15 $NAME $MODEL_FILE 0 174 png






