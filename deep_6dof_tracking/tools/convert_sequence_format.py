import os
import numpy as np
from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.sequence_loader import SequenceLoader


if __name__ == '__main__':
    path = "/media/ssd/eccv/choi"

    sequences = [os.path.join(path, x) for x in os.listdir(path)]

    for sequence in sequences:
        if not os.path.exists(os.path.join(sequence, "meta_data.json")):
            print("Convert {}".format(sequence))
            dataset = DeepTrackLoader(sequence)
            poses = []
            for frame, pose in dataset.data_pose:
                poses.append(pose.matrix.flatten())
            SequenceLoader.save(sequence, dataset.metadata, np.array(poses))
