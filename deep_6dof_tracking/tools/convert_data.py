import json
import os
import numpy as np
from deep_6dof_tracking.utils.transform import Transform
import deep_6dof_tracking.utils.angles as ea


def from_parameters(x, y, z, euler_x, euler_y, euler_z, is_degree=False):
    ret = Transform()
    ret.set_translation(x, y, z)
    ret.matrix[0:3, 0:3] = ea.euler2mat(euler_x, euler_y, euler_z)
    return ret

if __name__ == '__main__':
    input_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/choi_christensen/frame_synth_tide_kitchen"

    data = json.load(open(os.path.join(input_path, "viewpoints.json"), "r"))

    index = list(data.keys())
    index.sort()
    poses = np.zeros((len(index)-1, 16))

    for i in index:
        params = np.zeros(6)
        if "vector" in data[i]:
            params_dict = data[i]["vector"]
            for j in range(6):
                params[j] = params_dict[str(j)]
            pose = from_parameters(*params)
            poses[int(i), :] = pose.matrix.flatten()

    np.save(os.path.join(input_path, "poses.npy"), poses)
