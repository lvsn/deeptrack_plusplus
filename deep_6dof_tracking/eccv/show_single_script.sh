object=clock
sequence=${object}_motion_rotation


python3 show_prediction.py --sequence /media/ssd/eccv/Sequences/final_sequences/$sequence \
                                   --predictions /media/ssd/eccv/Results/comparison_results/general/$sequence\
                                   --model /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/${object} \
                                   --save_frames /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/to_delete \
                                   --color P --brightness 75 --show_gt --show