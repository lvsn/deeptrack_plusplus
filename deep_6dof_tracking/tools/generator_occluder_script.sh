CODE_PATH="`( pwd )`"
SAVE_PATH=/media/ssd/deeptracking

NAME=dragon
DATA_PATH=$SAVE_PATH/to_delete
mkdir $DATA_PATH
python3 dataset_generator.py -o $DATA_PATH -m model_configs/${NAME}.yml \
    -s 25000 --show -t 0.02 -r 20 -e 224 --saveformat numpy --debug
