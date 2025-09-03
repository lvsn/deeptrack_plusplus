#!/usr/bin/env bash
CODE_PATH="`( pwd )`"
REAL_PATH=/media/ssd/eccv/training_data

STORING_PATH=/home/mathieu/Dataset/eccv

timestamp () {
    date +"%y-%m-%d_%Hh%Mm%Ss"
}
LOG_FILE=$SAVE_PATH/$(timestamp)_generate.log
echo Writing logs in ${LOG_FILE}

generate () {
    if [ -f ${STORING_PATH}/$4.tar.gz ]
    then
        echo "Skip $4"
    else
        echo "Running $4"
        echo Start $4 $(timestamp) >> $LOG_FILE
        mkdir $DATA_PATH
        python3 dataset_generator_from_raw.py -o $1/train -m model_configs/$5 -s 30000 --show \
            -t $2 -r $3 --boundingbox $6 -e $7 --saveformat $8 --real_path $9/train/${10} --random_sampling
        #python3 dataset_generator_from_raw.py -o $1/valid -m model_configs/$5 -s 5000 --show \
        #    -t $2 -r $3 --boundingbox $6 -e $7 --saveformat $8 --real_path $9/valid/${10} --random_sampling
        cd $STORING_PATH
        tar -czf $4.tar.gz $4
        mv $4.tar.gz $STORING_PATH
        rm -r $4/
        cd $CODE_PATH
        echo Stop $4 $(timestamp) >> $LOG_FILE
    fi
}


MODEL_FILE=cookiejar.yml
OBJECT=cookiejar_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=cookiejar.yml
OBJECT=cookiejar_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT

MODEL_FILE=dog.yml
OBJECT=dog_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=dog.yml
OBJECT=dog_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT

MODEL_FILE=kinect.yml
OBJECT=kinect_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=kinect.yml
OBJECT=kinect_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT

MODEL_FILE=lego.yml
OBJECT=lego_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=lego.yml
OBJECT=lego_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT

MODEL_FILE=walkman.yml
OBJECT=walkman_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=walkman.yml
OBJECT=walkman_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT

MODEL_FILE=wateringcan.yml
OBJECT=wateringcan_top
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


MODEL_FILE=wateringcan.yml
OBJECT=wateringcan_bottom
NAME=${OBJECT}
DATA_PATH=$STORING_PATH/${NAME}
generate $DATA_PATH 0.03 20 $NAME $MODEL_FILE 10 150 numpy $REAL_PATH $OBJECT


