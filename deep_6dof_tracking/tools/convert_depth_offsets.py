import os
import numpy as np
from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.sequence_loader import SequenceLoader
from tqdm import tqdm


if __name__ == '__main__':
    path = "/media/ssd/eccv/sequences_keep"
    #keep_list = ["clock_occlusion_h_15", "clock_occlusion_h_30", "clock_occlusion_h_45", "clock_occlusion_h_60", "clock_occlusion_0"]

    sequences = [os.path.join(path, x) for x in os.listdir(path)]

    for sequence in sequences:
        #do_process = False
        #for keep in keep_list:
        #    if keep in sequence:
        #        do_process = True
        #if not do_process:
        #    print("Skip {}".format(sequence))
        #    continue
        print("Convert {}".format(sequence))
        dataset = SequenceLoader(sequence)
        poses = []
        for frame, pose in tqdm(dataset.data_pose):
            rgb, depth = frame.get_rgb_depth(sequence)
            #depth += 24
            #depth[depth == 24] = 0
            frame.depth = depth
            frame.rgb = rgb
            frame.dump(sequence)
