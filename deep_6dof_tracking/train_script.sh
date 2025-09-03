

STORAGE_PATH=/root/dataset_raid/deeptracking
DATA_PATH=/root/dataset_ssd
OUTPUT_PATH=/root/outputs/tracking
BACKGROUND_PATH=/root/dataset_raid/SUN3D
OCCLUDER_PATH=/root/dataset_raid/deeptracking/hand


train () {
        echo "Running $1"
        #cd tools
        #python3 dataset_mean.py -d $DATA_PATH/$1 -b $BACKGROUND_PATH -r $OCCLUDER_PATH
        #cd ..
        python3 train.py -e 20 -s 32 -i 1 -n -1 \
                --output $OUTPUT_PATH/$1_$3 \
                --dataset $DATA_PATH/$1 \
                --background $BACKGROUND_PATH \
                --occluder $OCCLUDER_PATH \
                --architecture $2
        cp train_script.sh $OUTPUT_PATH/$1
        echo removed $1
}

finetune() {
	echo "Running $1 Finetune"
	python3 train.py -e 20 -s 32 -i 1 -n -1 \
		--output $OUTPUT_PATH/$1_$3 \
		--dataset $DATA_PATH/$1 \
		--background $BACKGROUND_PATH \
		--occluder $OCCLUDER_PATH \
		--architecture $2 --phase 1 -l 0.0001 --finetune $OUTPUT_PATH/$1_split0_iccv/model_best.pth.tar
	cp train_script.sh $OUTPUT_PATH/$1
}

#cd tools
#python3 dataset_mean.py -d $DATA_PATH/generic -b $BACKGROUND_PATH -r $OCCLUDER_PATH
#cd ..

#train generic res wacv_rgb

train generic res_moddropout wacv_moddrop_donly_174_0
train generic res_rgb wacv_rgb_donly_174_0

train generic res_moddropout wacv_moddrop_donly_174_1
train generic res_rgb wacv_rgb_donly_174_1

#train generic res_moddropout wacv_moddrop_174_2
#train generic res_rgb wacv_rgb_174_2

#train generic res_moddropout wacv_moddrop_174_4
#train generic res_rgb wacv_rgb_174_4
#train generic res_moddropout wacv_moddrop_1
#train generic res_rgb wacv_rgb_1

#train generic res_moddropout wacv_moddrop_2
#train generic res_rgb wacv_rgb_2

#train generic res_moddropout wacv_moddrop_3
#train generic res_rgb wacv_rgb_3

#train generic res_moddropout wacv_moddrop_4
#train generic res_rgb wacv_rgb_4

#train kinect_box res_consistence consistence_iccv
#train kinect_box res iccv
#train ${dataset} res_moddropout moddrop_iccv
#train ${dataset} res_split split0_iccv
