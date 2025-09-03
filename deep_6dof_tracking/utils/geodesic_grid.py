"""
    Geodesic grid (as a pointcloud) that can be refined

    date : 2016-03-02
"""
# Credit to Andreas Kahler for its tutorial

import math
import deep_6dof_tracking.utils.pointcloud as pc
import numpy as np


class GeodesicGrid:
    def __init__(self):
        vertex = GeodesicGrid.get_icosahedron_vertex()
        self.cloud = pc.PointCloud.XYZ()
        self.cloud.resize(len(vertex))
        self.cloud["XYZ"] = vertex
        self.faces = GeodesicGrid.get_icosahedron_faces()
        self.vertex_cache = {}

    def insert_middle_point_index(self, index1, index2):
        """
        Check the cache if this middle point has already been inserted, if not, add it to the vertex list
        :param index1:
        :param index2:
        :return: index of the new vertex
        """
        key = self._get_unique_key(index1, index2)
        # Avoid creating a new vertex if it is already here
        if key in self.vertex_cache:
            return self.vertex_cache[key]

        point1 = self.cloud["XYZ"][index1]
        point2 = self.cloud["XYZ"][index2]
        middle = (point1 + point2) / 2
        length = math.sqrt(np.sum(middle ** 2))
        new_vertex = middle / length

        new_index = self.cloud.width
        self.cloud.add_vertex(XYZ=new_vertex)
        self.vertex_cache[key] = new_index
        return new_index

    def _get_unique_key(self, index1, index2):
        first_is_smaller = index1 < index2
        smaller_index = index1 if first_is_smaller else index2
        greater_index = index2 if first_is_smaller else index1
        key = str(smaller_index) + str(greater_index)
        return key

    def refine_icoshpere(self, it):
        """
        Divide the geodesic grid by two and make sure all points are projected to sphere of radius 1
        Updates the faces
        :param it:
        :return:
        """
        for i in range(it):
            self.vertex_cache.clear()
            ico_new_faces = None
            for face in self.faces:
                new_vertex1 = self.insert_middle_point_index(face[0], face[1])
                new_vertex2 = self.insert_middle_point_index(face[1], face[2])
                new_vertex3 = self.insert_middle_point_index(face[2], face[0])

                new_faces = np.array([
                    [face[0], new_vertex1, new_vertex3],
                    [face[1], new_vertex2, new_vertex1],
                    [face[2], new_vertex3, new_vertex2],
                    [new_vertex1, new_vertex2, new_vertex3]
                ])
                if ico_new_faces is not None:
                    ico_new_faces = np.vstack((ico_new_faces, new_faces))
                else:
                    ico_new_faces = new_faces
            self.faces = ico_new_faces

    def get_neighbors(self, vertex, angle_tresh):
        cloud = self.cloud["XYZ"]
        v = GeodesicGrid.unit_vector(vertex)
        indexes = []
        for i, point in enumerate(cloud):
            u = GeodesicGrid.unit_vector(point)
            angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            if math.degrees(angle) <= angle_tresh:
                indexes.append(i)
        return indexes

    @staticmethod
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def golden_ratio():
        return (1 + math.sqrt(5)) / 2

    @staticmethod
    def get_icosahedron_vertex():
        t = GeodesicGrid.golden_ratio()
        ico = np.array([
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],

            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],

            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1]
        ])

        length = np.array([np.sqrt(np.sum(ico ** 2, axis=1))]).T
        return np.divide(ico, length)

    @staticmethod
    def get_icosahedron_faces():
        faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],

            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],

            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],

            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ])
        return faces


if __name__ == "__main__":
    grid = GeodesicGrid()
    grid.refine_icoshpere(2)
    print('yo')
    v = grid.cloud.vertex['XYZ']
    print(np.all(v == [0,0,0], axis=1))
    v = v[~np.all(v == [0,0,0], axis=1)]
    f = grid.faces
    np.set_printoptions(edgeitems=100)
    print(v)
    print(f)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(len(v))

    # Plot the vertices
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], color='r')

    # Create a Poly3DCollection for the faces
    poly3d = [v[face] for face in f]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=1))

    # Set axis limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')

    # plt.show()

    # from uniform_sphere_sampler import UniformSphereSampler
    # sampler = UniformSphereSampler()
    # r = sampler.get_random()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pts = []
    from transform import Transform
    for i, j in enumerate(v):
        color='r'
        if i == 25 or i == 28:
            color = 'b'
            print('here')
            view = Transform.lookAt(j, np.zeros(3), np.array([0, 1, 0]))
        else:
            view = Transform.lookAt(j, np.zeros(3), np.array([0, 0, 1]))
        if np.any(view.to_parameters() == np.nan) or 'nan' in str(view):
            print('nan at ', i)
            print(view)
        pt1 = view.matrix@np.array([0, 0, 1, 1])
        pt2 = view.matrix@np.array([0, 0, 0, 1])
        if i == 25 or i == 28:
            print(pt1, pt2)
        if pt1[0] in [-1, 1]:
            print(i)
        # ax.scatter(pt[0], pt[1], pt[2], color='r')
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    plt.show()


    print('bleh')
    