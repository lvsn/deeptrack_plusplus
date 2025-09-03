import tarfile
import numpy as np
import time
import os
import json

from PIL import Image

if __name__ == '__main__':
    path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset"
    path_tar = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/SUN3D.tar"

    file_name_rgb = "SUN3D/home_puigpunyent_scan4_2012_dec_23/25560.png"
    file_name_depth = "SUN3D/home_puigpunyent_scan4_2012_dec_23/25560d.png"

    t = tarfile.open(path_tar, 'r')

    time_start = time.time()
    f = t.extractfile(file_name_rgb)
    rgb = np.array(Image.open(t.extractfile(file_name_rgb)))
    depth = np.array(Image.open(t.extractfile(file_name_depth))).astype(np.uint16)

    print("get files time {}".format(time.time() - time_start))
    print(np.mean(rgb))
    print(np.mean(depth))

    time_start = time.time()
    rgb = np.array(Image.open(os.path.join(path, file_name_rgb)))
    depth = np.array(Image.open(os.path.join(path, file_name_depth))).astype(np.uint16)

    print("get files time {}".format(time.time() - time_start))
    print(np.mean(rgb))
    print(np.mean(depth))

    path = "/media/ssd/deeptracking/dragon_norm_4"
    path_tar = "/media/ssd/deeptracking/valid.tar"

    t = tarfile.open(path_tar, 'r')

    file_name_json = "valid/viewpoints.json"

    file_name_numpy = "valid/159.npy"

    time_start = time.time()
    f = t.extractfile(file_name_json)
    data = np.load(f)

    print("get files time {}".format(time.time() - time_start))
    print(data)

    time_start = time.time()
    data = np.load(os.path.join(path, file_name_numpy))
    print("get files time {}".format(time.time() - time_start))
    print(data)

    time_start = time.time()
    f = t.extractfile(file_name_json)
    data = json.load(f.read())

    print("get files time {}".format(time.time() - time_start))
    print(data)

    time_start = time.time()
    data = json.load(os.path.join(path, file_name_json))
    print("get files time {}".format(time.time() - time_start))
    print(data)
