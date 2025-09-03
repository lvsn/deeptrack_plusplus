"""
Utility to read rgbd files in folders ( used for background loading)
"""

import numpy as np
import os
import random
from PIL import Image, ImageFile
import cv2


class RGBDDataset:
    def __init__(self, path, preload=False):
        self.do_preload = preload
        self.preloaded = []
        self.indexes = {}
        self.indexes_list = []
        self.path = path
        self.index_frames_()

    def index_frames_(self):
        dirs = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        for dir in dirs:
            dir_path = os.path.join(self.path, dir)
            rgbPath = os.path.join(dir_path, 'images')
            files = [os.path.splitext(f)[0] for f in os.listdir(rgbPath) if os.path.splitext(f)[1] == ".jpg"]
            files.sort()
            files = [str(f) for f in files]
            self.indexes[dir] = files
            for file in files:
                self.indexes_list.append((dir, file))
                if self.do_preload:
                    color, depth = self.load_sample(dir, file)
                    self.preloaded.append((color, depth))

    def load_sample(self, dir, img):
        directory = os.path.join(self.path, dir)
        image_dir = os.path.join(directory, 'images')
        depth_dir = os.path.join(directory, 'depth')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        color = np.array(Image.open(os.path.join(image_dir, img + ".jpg")))
        #color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = np.array(Image.open(os.path.join(depth_dir, img + ".png"))).astype(np.uint16)
        return color, depth

    def load_random_sample(self, index=None):
        rand_int = index
        if index is None:
            rand_int = random.randint(0, len(self.indexes_list) - 1)
        if self.do_preload:
            color, depth = self.preloaded[rand_int]
        else:
            dir, file = self.indexes_list[rand_int]
            color, depth = self.load_sample(dir, file)
        return color, depth

    def load_random_image(self, size):
        color, depth = self.load_random_sample()
        x, y = RGBDDataset.get_random_crop(color.shape[0], color.shape[1], size)
        color = color[x:x+size, y:y+size, :]
        depth = depth[x:x+size, y:y+size]
        return color, depth

    def load_random_image_full_res(self, size, index=None):
        color, depth = self.load_random_sample(index=index)
        x, y = RGBDDataset.get_random_crop_full_res(color.shape[0], color.shape[1], size)
        color = color[x:x+size[0], y:y+size[1], :]
        depth = depth[x:x+size[0], y:y+size[1]]
        return color, depth
    
    def load_image(self, size, id):
        color, depth = self.load_random_sample(index=id)
        x, y = RGBDDataset.get_random_crop(color.shape[0], color.shape[1], size)
        color = color[x:x+size, y:y+size, :]
        depth = depth[x:x+size, y:y+size]
        return color, depth


    def load_random_sequence(self):
        dir = random.choice(list(self.indexes.keys()))
        sequence = []
        for i in range(len(self.indexes[dir])):
            sequence.append(self.load_sample(dir, str(i)))
        return sequence

    @staticmethod
    def get_random_crop(w, h, size):
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return x, y
    
    @staticmethod
    def get_random_crop_full_res(w, h, size):
        x = random.randint(0, w - size[0])
        y = random.randint(0, h - size[1])
        return x, y

