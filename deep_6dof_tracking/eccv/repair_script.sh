#!/usr/bin/env bash

object=cookiejar
seqence_path=/media/ssd/eccv/Sequences/sequences/${object}_fix_occluded_3

python3 repair_prediction.py --sequence ${seqence_path} \
                        --model /media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models/${object}
