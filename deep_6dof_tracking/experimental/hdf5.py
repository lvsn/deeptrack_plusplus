import h5py
import numpy as np
import os

if __name__ == '__main__':

    path = "test"

    # write data
    rgb = np.zeros((150, 150, 3), np.uint8)
    depth = np.zeros((150, 150), np.uint16)
    id = 0
    f = h5py.File(os.path.join(path, "data.hdf5"), "w")
    f.create_dataset("{}".format(id), data=rgb)
    f.create_dataset("{}d".format(id), data=depth)
    f.close()

    f = h5py.File(os.path.join(path, "data_compressed.hdf5"), "w")
    f.create_dataset("{}".format(id), data=rgb, compression="gzip")
    f.create_dataset("{}d".format(id), data=depth, compression="gzip")
    f.close()

    f = h5py.File(os.path.join(path, "data.hdf5"), "r")
    data_keys = list(f.keys())
    rgb = f[data_keys[0]][:]
    depth = f[data_keys[1]][:]
    f.close()

    f = h5py.File(os.path.join(path, "data_compressed.hdf5"), "r")
    data_keys = list(f.keys())
    rgb = f[data_keys[0]][:]
    depth = f[data_keys[1]][:]
    f.close()
