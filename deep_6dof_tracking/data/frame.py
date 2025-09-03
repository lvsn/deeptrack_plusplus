"""
    Keeps a reference to a rgb and depth images (given an id). keeps data on disk or ram depending on need

    date : 2017-23-03
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import os
import numpy as np
from PIL import Image


class Frame(object):
    def __init__(self, rgb, depth, id, depthonly=False, mask=None):
        self.rgb = rgb
        self.depth = depth
        self.id = id
        self.depthonly = depthonly
        self.mask = mask

    def is_on_disk(self):
        return self.rgb is None and self.depth is None

    def exists(self, path):
        return os.path.exists(os.path.join(path, '{}.png').format(self.id)) and \
               os.path.exists(os.path.join(path, '{}d.png').format(self.id))

    def get_rgb_depth(self, path, keep_in_ram=False):
        """
        getter to images, if keep in ram is false, the reference wont be kept by this object
        :param path:
        :param keep_in_ram:
        :return:
        """
        if self.is_on_disk():
            self.load(path)
        rgb = self.rgb
        depth = self.depth
        if not keep_in_ram:
            self.clear_image()
        return rgb, depth

    def clear_image(self):
        self.rgb = None
        self.depth = None

    def dump(self, path, pad_name=False):
        if not self.is_on_disk():
            import cv2
            if pad_name:
                fname = '{:05d}.png'.format(int(self.id))
                cv2.imwrite(os.path.join(path, 'rgb', fname), self.rgb[:, :, ::-1])
                cv2.imwrite(os.path.join(path, 'depth', fname), self.depth.astype(np.uint16))
                if self.mask is not None:
                    cv2.imwrite(os.path.join(path, 'masks', fname), self.mask)

            else:
                if not self.depthonly:
                    cv2.imwrite(os.path.join(path, '{}.png').format(self.id), self.rgb[:, :, ::-1])
                    if self.mask is not None:
                        cv2.imwrite(os.path.join(path, '{}m.png').format(self.id), self.mask)
                cv2.imwrite(os.path.join(path, '{}d.png').format(self.id), self.depth.astype(np.uint16))
                self.clear_image()

    def load(self, path):
        self.rgb = np.array(Image.open(os.path.join(path, self.id + ".png")))
        self.depth = np.array(Image.open(os.path.join(path, self.id + "d.png"))).astype(np.uint16)


class FrameNumpy(Frame):
    def __init__(self, rgb, depth, id, mask=None, depthonly=False):
        super(FrameNumpy, self).__init__(rgb, depth, id, depthonly=depthonly)

    def dump(self, path):
        if not self.is_on_disk():
            depth8 = self.numpy_int16_to_uint8(self.depth)
            if not self.depthonly:
                frame = np.concatenate((self.rgb, depth8), axis=2)
                np.save(os.path.join(path, self.id), frame)
            else:
                np.save(os.path.join(path, self.id), depth8)
            self.clear_image()

    def load(self, path):
        frame = np.load(os.path.join(path, "{}.npy".format(self.id)))
        self.depth = self.numpy_uint8_to_int16(frame[:, :, 3:])
        self.rgb = frame[:, :, 0:3]

    @staticmethod
    def numpy_int16_to_uint8(depth):
        x, y = depth.shape
        out = np.ndarray((x, y, 2), dtype=np.uint8)
        out[:, :, 0] = np.right_shift(depth, 8)
        out[:, :, 1] = depth.astype(np.uint8)
        return out

    @staticmethod
    def numpy_uint8_to_int16(depth8):
        x, y, c = depth8.shape
        out = np.ndarray((x, y), dtype=np.int16)
        out[:, :] = depth8[:, :, 0]
        out = np.left_shift(out, 8)
        out[:, :] += depth8[:, :, 1]
        return out


class FrameHdf5(Frame):
    file_write_dict = {}
    file_read_dict = {}
    def __init__(self, rgb, depth, id, mask=None):
        super(FrameHdf5, self).__init__(rgb, depth, id)

    def dump(self, path):
        if not self.is_on_disk():
            self.create_file(path, 'w')
            FrameHdf5.file_write_dict[path].create_dataset(self.id, data=self.rgb)
            FrameHdf5.file_write_dict[path].create_dataset(self.id+"d", data=self.depth)
            self.clear_image()

    def load(self, path):
        self.create_file(path, 'r')
        self.rgb = FrameHdf5.file_read_dict[path][self.id][:]
        self.depth = FrameHdf5.file_read_dict[path][self.id][:]

    # hack.. the class can create 1 file per folder...
    def create_file(self, path, rw):
        import h5py
        if "r" in rw:
            if path not in FrameHdf5.file_read_dict.keys():
                f = h5py.File(os.path.join(path, "data.hdf5"), rw)
                FrameHdf5.file_read_dict[path] = f
        else:
            if path not in FrameHdf5.file_write_dict.keys():
                f = h5py.File(os.path.join(path, "data.hdf5"), rw)
                FrameHdf5.file_write_dict[path] = f