"""
    PointCloud utility with numpy backend
    todo: simplify/remove this...

    date : 2016-03-01
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import numpy as np
import copy
import scipy.spatial
from PIL import Image
from scipy.spatial import ConvexHull, distance_matrix


class PointCloudChannel:
    def __init__(self):
        self.size = 0
        self.tags = {}

    def add_tag(self, name, dtype):
        if dtype in self.tags:
            self.tags[dtype].append(name)
        else:
            self.tags[dtype] = [name]

    def set_size(self, size):
        self.size = size

    def get_arrays(self):
        ret = {}
        for key in self.tags:
            block_name = self.get_block_name(self.tags[key])
            ret[block_name] = np.zeros((self.size, len(self.tags[key])), dtype=key)
        return ret

    def get_block_names(self):
        ret = []
        for key in self.tags:
            ret.append(self.get_block_name(self.tags[key]))
        return ret

    def get_block_name(self, tags):
        name = ""
        for tag in tags:
            name += tag[0].upper()
        return name

    def __iter__(self):
        return iter(self.get_block_names())

    def __contains__(self, item):
        return item in self.tags

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.tags) + " Of Size " + str(self.size)


class PointCloud:
    """
        PointCloud is a data structure that can contain multiple modalitys. The user may define its properties with
        a list of Channels which is a description of data type and structure 1 character structure names
        channels_list (list of Channels) : each channels will have memory allocation
        block_size (int) : initial memory allocated
    """

    def __init__(self, channels_dict):
        assert "vertex" in channels_dict, "PointCloud needs a vertex channel"
        self.width = 0
        self.height = 1
        self.vertex_reserved_size = 1000
        self.vertex = {}
        self.camera = None
        self.is_organized = False
        self._setup_vertex(channels_dict["vertex"])
        self.channel_dict = {"vertex": channels_dict["vertex"]}
        self.texture = None

    def copy(self):
        ret = PointCloud(self.channel_dict)
        ret.width = self.width
        ret.height = self.height
        ret.vertex_reserved_size = self.vertex_reserved_size
        ret.vertex = copy.deepcopy(self.vertex)
        ret.camera = self.camera
        ret.is_organized = self.is_organized
        return ret

    def _setup_vertex(self, channel):
        self.vertex = channel.get_arrays()
        self._reserve_vertex_size(channel.size)
        self.width = channel.size

    def organize(self, shape):
        assert shape[0] * shape[1] == self.width * self.height
        self.is_organized = not shape[1] == 1
        self.height, self.width = shape

    def unorganize(self):
        self.is_organized = False
        self.width *= self.height
        self.height = 1

    def change_type(self, channel, dtype):
        self.vertex[channel] = self.vertex[channel].astype(dtype)

    def set_texture(self, path):
        self.texture = np.array(Image.open(path)).astype(np.uint8)

    @property
    def size(self):
        return self.width * self.height

    @staticmethod
    def XYZ():
        channel = PointCloudChannel()
        channel.add_tag('X', np.float32)
        channel.add_tag('Y', np.float32)
        channel.add_tag('Z', np.float32)
        return PointCloud({'vertex': channel})

    @staticmethod
    def XYZRGB():
        channel = PointCloudChannel()
        channel.add_tag('X', np.float32)
        channel.add_tag('Y', np.float32)
        channel.add_tag('Z', np.float32)
        channel.add_tag('R', np.int8)
        channel.add_tag('G', np.int8)
        channel.add_tag('B', np.int8)
        return PointCloud({'vertex': channel})

    def __getitem__(self, key):
        for block in self.vertex:
            start, end = self._name_slice(key, block)
            if start != -1:
                return self.vertex[block][:self.size, start:end]
        raise KeyError("Invalid Key : " + key)

    def __setitem__(self, key, value):
        for block in self.vertex:
            start, end = self._name_slice(key, block)
            if start != -1:
                data_shape = self.vertex[block][:self.size, start:end].shape
                if data_shape != value.shape:
                    raise IndexError("Shape of matrix not compatible :" \
                                     + key + " = " + str(data_shape) + " and given = " + str(value.shape))
                self.vertex[block][:self.size, start:end] = value
                return
        raise KeyError("Invalid Key : " + key)

    def _name_slice(self, name, fullname):
        length = len(name)
        start_index = fullname.find(name)
        return start_index, start_index + length

    def __contains__(self, item):
        for block in self.vertex:
            if item in block:
                return True
        return False

    def add_vertex(self, **kwargs):
        """
        Add data to the end of the buffer, user may pass a dictionnary containing data for each data labels
        :param kwargs:
        :return:
        """
        if self.size + 1 > self.vertex_reserved_size:
            self._reserve_vertex_size()
        for arg in self.channel_dict["vertex"]:
            if arg in kwargs:
                self.vertex[arg][self.size, :] = kwargs[arg]
            else:
                raise KeyError("channel " + str(arg) + " has no input data.")
        self.width += 1  # todo watch out it breaks cloud's organization...

    def _reserve_vertex_size(self, minimum=1000):
        if self.vertex_reserved_size < minimum:
            self.vertex_reserved_size = minimum
        else:
            self.vertex_reserved_size *= 2
        for block in self.vertex:
            new_shape = list(self.vertex[block].shape)
            new_shape[0] = self.vertex_reserved_size - new_shape[0]
            self.vertex[block] = np.vstack((self.vertex[block], np.zeros(new_shape, dtype=self.vertex[block].dtype)))

    def remove(self, index):
        """
        remove whole vertex from numpy array efficiently, decrease size by 1
        :param index:
        :return:
        """
        for name in self.vertex:
            self.vertex[name][index:, :] = np.roll(self.vertex[name][index:, :], -1, axis=0)
        self.width -= 1

    def resize(self, value):
        while self.vertex_reserved_size <= self.width + value:
            self._reserve_vertex_size()  # todo check error here!
        self.width = value

    def transform(self, transform):
        """
        Apply transformation matrix on point cloud
        :param transform: 4X4 matrix containing affine transform
        :param pointcloud: reference to PointCloud object that will be modified
        :return:
        """
        assert transform.shape == (4, 4)  # Make sure it is a 4x4 affine transform matrix
        self["XYZ"] = transform.dot(self["XYZ"])

    def convex_hull_vertices(self):
        return scipy.spatial.ConvexHull(self["XYZ"])

    def diameter(self):
        cloud = self['XYZ']
        if self.is_organized:
            cloud = cloud[~np.all(cloud == 0, axis=1)]
        return scipy.spatial.distance.pdist(cloud).max()

    def __str__(self):
        ret = "Size = " + str((self.width, self.height)) + "\n"
        for channel in self.channel_dict:
            ret += str(channel) + "\n"
        return ret


def maximum_width(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    distances = distance_matrix(hull_points, hull_points)
    max_width = np.max(distances)
    return max_width