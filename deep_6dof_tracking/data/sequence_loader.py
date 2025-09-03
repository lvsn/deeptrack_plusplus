import os
import json
import numpy as np

from tqdm import tqdm
from PIL import Image

from deep_6dof_tracking.data.frame import FrameNumpy, Frame, FrameHdf5
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform


class SequenceLoader:
    def __init__(self, root):
        self.root = root
        self.data_pose = []
        self.load(self.root)

    def load(self, path):
        """
        Load a viewpoints.json to dataset's structure
        Todo: datastructure should be more similar to json structure...
        :return: return false if the dataset is empty.
        """
        # Load viewpoints file and camera file
        with open(os.path.join(path, "meta_data.json")) as data_file:
            data = json.load(data_file)
        self.camera = Camera.load_from_json(path)
        self.metadata = data["metaData"]
        self.set_save_type(self.metadata["save_type"])
        self.poses = np.load(os.path.join(os.path.join(path, "poses.npy")))
        # todo this is not clean!i
        for i, pose in enumerate(self.poses):
            pose = Transform.from_matrix(pose.reshape(4, 4))
            self.add_pose(None, None, pose)
        return self.data_pose

    @staticmethod
    def save(path, metadata, poses):
        viewpoints = {}
        viewpoints["metaData"] = metadata
        np.save(os.path.join(path, "poses.npy"), np.array(poses))
        with open(os.path.join(path, "meta_data.json"), 'w') as outfile:
            json.dump(viewpoints, outfile)

    def load_image(self, index):
        frame, pose = self.data_pose[index]
        rgb, depth = frame.get_rgb_depth(self.root)
        return rgb, depth, pose

    def load_pair(self, index, pair_id):
        frame, pose = self.data_pair[int(index)][pair_id]
        rgb, depth = frame.get_rgb_depth(self.root)
        return rgb, depth, pose

    def size(self):
        return len(self.data_pose)

    def set_save_type(self, frame_class):
        if frame_class == "numpy":
            self.frame_class = FrameNumpy
        elif frame_class == "hdf5":
            self.frame_class = FrameHdf5
        else:
            self.frame_class = Frame

    def add_pose(self, rgb, depth, pose):
        index = self.size()
        frame = self.frame_class(rgb, depth, str(index))
        self.data_pose.append([frame, pose])
        return index

    def pair_size(self, id):
        id_int = int(id)
        if id_int not in self.data_pair:
            return 0
        else:
            return len(self.data_pair[id_int])

    def add_pair(self, rgb, depth, pose, id):
        id_int = int(id)
        if id_int >= len(self.data_pose):
            raise IndexError("impossible to add pair if pose does not exists")
        if id_int in self.data_pair:
            frame = self.frame_class(rgb, depth, "{}n{}".format(id_int, len(self.data_pair[id_int]) - 1))
            self.data_pair[id_int].append((frame, pose))
        else:
            frame = self.frame_class(rgb, depth, "{}n0".format(id_int))
            self.data_pair[id_int] = [(frame, pose)]

    def dump_images_on_disk(self, verbose=False):
        """
        Unload all images data from ram and save them to the dataset's path ( can be reloaded with load_from_disk())
        :return:
        """
        print("[INFO]: Dump image on disk")
        for frame, pose in tqdm(self.data_pose):
            if verbose:
                print("Save frame {}".format(frame.id))
            if int(frame.id) in self.data_pair:
                for pair_frame, pair_pose in self.data_pair[int(frame.id)]:
                    pair_frame.dump(self.root)
            frame.dump(self.root)

    @staticmethod
    def insert_pose_in_dict(dict, key, item):
        params = {}
        for i, param in enumerate(item.to_parameters()):
            params[str(i)] = str(param)
        dict[key] = {"vector": params}


class Sequence():
    def __init__(self, root, depthonly=False, preload=True):
        self.root = root
        self.rgb = []
        self.depth = []
        self.preload = preload
        self.depthonly = depthonly
        if preload:
            self.load(self.root)

    def load(self, path):
        file_names = [filename for filename in os.listdir(path) if filename.endswith('.png')]
        depth_file_names = [filename for filename in file_names if 'd' in filename]
        other_file_names = [filename for filename in file_names if 'd' not in filename and 'm' not in filename]
        depth_file_names = sorted(depth_file_names, key=lambda x: int(x.split('d')[0]))
        other_file_names = sorted(other_file_names, key=lambda x: int(x.split('.')[0]))

        for filename in depth_file_names:
            image_path = os.path.join(path, filename)
            img = Image.open(image_path)
            self.depth.append(np.array(img))
        for filename in other_file_names:
            if self.depthonly:
                self.rgb.append(None)
            else:
                image_path = os.path.join(path, filename)
                img = Image.open(image_path)
                self.rgb.append(np.array(img)[:,:,:3])

    def size(self):
        return len(self.depth) if self.depthonly else len(self.rgb)
    
    def load_image(self, index):
        if self.preload:
            rgb = self.rgb[index] if not self.depthonly else None
            depth = self.depth[index]
            return rgb, depth
        else:
            # TODO
            raise NotImplementedError