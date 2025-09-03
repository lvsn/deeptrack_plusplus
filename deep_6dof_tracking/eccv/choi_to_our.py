import numpy as np
from deep_6dof_tracking.utils.transform import Transform

if __name__ == '__main__':
    file_path = "/home/mathieu/Dataset/synthetic_track/ground_truth/orange_juice_kitchen.motion"
    with open(file_path) as f:
        contents = f.readlines()

    contents = [[float(val) for val in x.split(" ")] for x in contents]
    scale = Transform.scale(1, -1, -1)
    inverse_contents = []
    for content in contents:
        t = Transform.from_matrix(np.array(content).reshape((4, 4)))
        t = scale.combine(t, copy=True)
        inverse_contents.append(t.matrix.flatten())

    np.save("/home/mathieu/Dataset/synthetic_track/ground_truth/poses.npy", np.array(inverse_contents))
    #for content in contents:
    #    print(content)