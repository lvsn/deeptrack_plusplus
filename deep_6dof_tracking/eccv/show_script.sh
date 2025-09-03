#!/usr/bin/env bash

model=general
color=P

output_path=/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/to_delete
prediction_path=/media/ssd/eccv/Results/comparison_results/${model}

#for object_path in ${prediction_path}/*; do
#    object=${object_path##*/}
#    for sequence_path in ${object_path}/*; do
#        sequence_name=${sequence_path##*/}
#        python3 show_prediction.py --sequence /media/ssd/eccv/Sequences/final_sequences/${object}_${sequence_name} \
#                                   --predictions ${object_path}/${sequence_name} \
#                                   --model /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/${object} \
#                                   --save_frames ${output_path}/${model}/${object}_${sequence_name} \
#                                   --color ${color} --brightness 75
#    done
#
#done

for sequence_path in ${prediction_path}/*; do
    sequence=${sequence_path##*/}
    object=${sequence%%_*}
    sequence_name=${sequence}
    save_path=${output_path}/${model}/${sequence_name}
    if [ -d ${save_path} ]
    then
        echo skip ${save_path}
    else
        echo process ${save_path}
        python3 show_prediction.py --sequence /media/ssd/eccv/Sequences/final_sequences/${sequence} \
                                   --predictions ${prediction_path}/${sequence} \
                                   --model /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/${object} \
                                   --save_frames ${save_path} \
                                   --color ${color} --brightness 75 --show_gt
    fi
done