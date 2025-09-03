"""
    minimalistic Opengl based rendering tool

    date : 2017-20-03
"""
__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import OpenGL.GL as gl
import vispy
from vispy import app, gloo

from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
import numpy as np
import os
import torch


class ModelRenderer2(app.Canvas):
    def __init__(self, model_path, shader_path, camera, window_sizes, backend="pyglet", model_scale=1, object_max_width=220):
        """
        each window size will setup one fpo
        :param model_path:
        :param shader_path:
        :param camera:
        :param window_size: list of size tuple [(height, width), (height2, width2) ... ]
        """
        #print(vispy.sys_info())
        app.use_app()  # Set backend
        app.Canvas.__init__(self, show=False, size=(camera.width, camera.height))

        fragment_code = open(os.path.join(shader_path, "fragment_light3.txt"), 'r').read()
        vertex_code = open(os.path.join(shader_path, "vertex_light3.txt"), 'r').read()

        self.model_3d = PlyParser(model_path)

        #gloo.gl.use_gl('gl2 debug')

        self.window_sizes = window_sizes
        self.camera = camera
        self.object_max_width = maximum_width(self.model_3d.get_vertex() * model_scale)*1000

        self.rgb = np.array([])
        self.depth = np.array([])

        # Create buffers
        vertices = self.model_3d.get_vertex() * model_scale
        faces = self.model_3d.get_faces()
        self.data = np.ones(vertices.shape[0], [('a_position', np.float32, 3),
                                           ('a_color', np.float32, 3),
                                           ('a_normal', np.float32, 3),
                                           ('a_ambiant_occlusion', np.float32, 3),
                                           ('a_texcoords', np.float32, 2)])
        self.data['a_position'] = vertices
        self.data['a_color'] = self.model_3d.get_vertex_color()
        self.data['a_color'] /= 255.
        self.data['a_normal'] = self.model_3d.get_vertex_normals()
        self.data['a_normal'] = self.data['a_normal'] / np.linalg.norm(self.data['a_normal'], axis=1)[:, np.newaxis]
        self.faces = self.model_3d.data["face"].data['vertex_indices']

        # Calculate face areas
        face_areas = []
        for face in self.faces:
            v1, v2, v3 = self.data['a_position'][face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            face_areas.append(area)

        # Total surface area
        total_area = sum(face_areas)
        self.face_areas = np.array(face_areas) / total_area

        # set white texture by default
        texture = np.ones((1, 1, 4), dtype=np.uint8)
        texture.fill(255)
        # else load the texture from the model
        try:
            self.data['a_texcoords'] = self.model_3d.get_texture_coord()
            texture = self.model_3d.get_texture()
            if texture is not None:
                texture = texture[::-1, :, :]
        except KeyError:
            pass

        self.vertex_buffer = gloo.VertexBuffer(self.data)
        self.index_buffer = gloo.IndexBuffer(faces.flatten().astype(np.uint32))

        self.program = gloo.Program(vertex_code, fragment_code)

        self.program.bind(self.vertex_buffer)
        self.program['tex'] = gloo.Texture2D(texture)

        self.program['shininess'] = [0]
        self.program['lightA_diffuse'] = [1, 1, 1]
        self.program['lightA_direction'] = [-1, -1, 1]
        self.program['lightA_specular'] = [1, 1, 1]

        self.program['ambientLightForce'] = [0.65, 0.65, 0.65]
        self.setup_camera(self.camera, 0, self.camera.width, self.camera.height, 0)

        # Frame buffer object
        self.fbos = []
        for window_size in self.window_sizes:
            shape = (window_size[1], window_size[0])
            self.fbos.append(gloo.FrameBuffer(gloo.Texture2D(shape=shape + (3,)), gloo.RenderBuffer(shape)))

    def get_random_pts_on_surface(self, num_points_to_sample):
        random_faces_indices = np.random.choice(len(self.faces), size=num_points_to_sample, p=self.face_areas)
        random_faces = self.faces[random_faces_indices]

        bary_coords = np.random.rand(num_points_to_sample, 3)
        bary_coords /= bary_coords.sum(axis=1)[:, None]

        first_elements = np.array([sub_arr[0] for sub_arr in random_faces])
        second_elements = np.array([sub_arr[1] for sub_arr in random_faces])
        third_elements = np.array([sub_arr[2] for sub_arr in random_faces])

        v1 = self.data['a_position'][first_elements]
        v2 = self.data['a_position'][second_elements]
        v3 = self.data['a_position'][third_elements]

        x = bary_coords[:,0] * v1[:,0] + bary_coords[:,1] * v2[:,0] + bary_coords[:,2] * v3[:,0]
        y = bary_coords[:,0] * v1[:,1] + bary_coords[:,1] * v2[:,1] + bary_coords[:,2] * v3[:,1]
        z = bary_coords[:,0] * v1[:,2] + bary_coords[:,1] * v2[:,2] + bary_coords[:,2] * v3[:,2]

        sampled_points = np.stack((x, y, z), axis=1)
        return sampled_points
    
    def get_random_pts_on_surface_batch(self, num_points_to_sample, batch_size):
        random_faces_indices = np.random.choice(len(self.faces), size=(batch_size, num_points_to_sample), p=self.face_areas)
        random_faces = self.faces[random_faces_indices]

        bary_coords = np.random.rand(batch_size, num_points_to_sample, 3)
        bary_coords /= bary_coords.sum(axis=2)[:, :, None]

        first_elements = np.array([[sub_arr[0] for sub_arr in random_faces[i]] for i in range(batch_size)])
        second_elements = np.array([[sub_arr[1] for sub_arr in random_faces[i]] for i in range(batch_size)])
        third_elements = np.array([[sub_arr[2] for sub_arr in random_faces[i]] for i in range(batch_size)])

        v1 = self.data['a_position'][first_elements]
        v2 = self.data['a_position'][second_elements]
        v3 = self.data['a_position'][third_elements]

        x = bary_coords[:,:,0] * v1[:,:,0] + bary_coords[:,:,1] * v2[:,:,0] + bary_coords[:,:,2] * v3[:,:,0]
        y = bary_coords[:,:,0] * v1[:,:,1] + bary_coords[:,:,1] * v2[:,:,1] + bary_coords[:,:,2] * v3[:,:,1]
        z = bary_coords[:,:,0] * v1[:,:,2] + bary_coords[:,:,1] * v2[:,:,2] + bary_coords[:,:,2] * v3[:,:,2]

        sampled_points = np.stack((x, y, z), axis=2)
        return sampled_points
    
    def pre_sample_points(self, num_points_to_sample, preload=False):
        samples = self.get_random_pts_on_surface(num_points_to_sample)
        if preload:
            self.samples = torch.from_numpy(samples).cuda().float()
        else:
            self.samples = samples

    def get_pre_sampled_points(self, num_points_to_sample, batch_size):
        batch_indices = torch.randint(0, 10000, (batch_size, num_points_to_sample))
        sampled_points = self.samples[batch_indices]
        return sampled_points

    def load_ambiant_occlusion_map(self, path):
        try:
            ao_model = PlyParser(path)
            self.data["a_ambiant_occlusion"] = ao_model.get_vertex_color()
            self.data["a_ambiant_occlusion"] /= 255
        except FileNotFoundError:
            print("[WARNING] ViewpointRender: ambiant occlusion file not found ... continue with basic render")

    def setup_camera(self, camera, left, right, bottom, top):
        self.near_plane = 0.01
        self.far_plane = 2.5

        # credit : http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        proj = np.array([[camera.focal_x, 0, -camera.center_x, 0],
                         [0, camera.focal_y, -camera.center_y, 0],
                         [0, 0, self.near_plane + self.far_plane, self.near_plane * self.far_plane],
                         [0, 0, -1, 0]])
        self.projection_matrix = self.orthographicMatrix(left,
                                                         right,
                                                         bottom,
                                                         top,
                                                         self.near_plane,
                                                         self.far_plane).dot(proj).T
        self.program['input_proj'] = self.projection_matrix

    @staticmethod
    def orthographicMatrix(left, right, bottom, top, near, far):
        right = float(right)
        left = float(left)
        top = float(top)
        bottom = float(bottom)
        mat = np.array([[2. / (right - left), 0, 0, -(right + left) / (right - left)],
                        [0, 2. / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                        [0, 0, 0, 1]], dtype=np.float32)
        return mat

    def on_draw(self, event, fbo_index):
        size = self.window_sizes[fbo_index]
        fbo = self.fbos[fbo_index]
        with fbo:
            gloo.set_state(depth_test=True)
            #gl.glEnable(gl.GL_LINE_SMOOTH)
            #gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gloo.set_cull_face('back')  # Back-facing polygons will be culled
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *size)
            self.program.draw('triangles', self.index_buffer)

            # Retrieve the contents of the FBO texture
            self.rgb = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            self.rgb = np.frombuffer(self.rgb, dtype=np.uint8).reshape((size[1], size[0], 3))
            self.depth = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            self.depth = self.depth.reshape(size[::-1])
            self.depth = self.gldepth_to_worlddepth(self.depth)

    def gldepth_to_worlddepth(self, frame):
        A = self.projection_matrix[2, 2]
        B = self.projection_matrix[3, 2]
        distance = B / (frame * -2.0 + 1.0 - A) * -1
        idx = distance[:, :] >= B / (A + 1)
        distance[idx] = 0
        return (distance * 1000).astype(np.uint16)

    def render_image(self, view_transform, fbo_index=0, light_direction=None, light_diffuse=None, ambiant_light=None, shininess=0):
        if light_direction is None:
            light_direction = np.array([0, 0, 1])
        if ambiant_light is None:
            ambiant_light = np.array([0.7, 0.7, 0.7])
        if light_diffuse is None:
            light_diffuse = np.array([0.5, 0.5, 0.5])

        #light_normal = np.ones(4)
        #light_normal[0:3] = light_direction
        #light_direction = np.dot(view_transform.transpose().inverse().matrix, light_normal)[:3]
        self.program["lightA_direction"] = light_direction
        self.program["ambientLightForce"] = ambiant_light
        self.program['lightA_diffuse'] = light_diffuse
        self.program['input_view'] = view_transform.matrix.T
        self.program['input_normal'] = view_transform.transpose().inverse().matrix.T
        self.program['shininess'] = shininess

        self.update()
        self.on_draw(None, fbo_index)
        return self.rgb, self.depth
    

if __name__ == "__main__":

    model_path = "models/chariot_mini/geometry.ply"

    from deep_6dof_tracking.utils.camera import Camera
    camera_path = "deep_6dof_tracking/data/sensors/camera_parameter_files/synthetic.json"
    camera = Camera.load_from_json(camera_path)

    shader_path = "deep_6dof_tracking/data/shaders"

    window_size = [camera.width, camera.height]

    mr = ModelRenderer2(model_path, shader_path, camera, [window_size, (174, 174)], object_max_width=734)
    mr.pre_sample_points(40000, preload=True)
    pts = mr.get_pre_sampled_points(1000, 16)

    #pts_cpy = pts.clone().cpu().numpy()
    pts_cpy = mr.samples.clone().cpu().numpy()
    import matplotlib.pyplot as plt
    for i in range(1):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal', 'box')
        # ax.scatter(pts_cpy[i, :, 0], pts_cpy[i, :, 1], pts_cpy[i, :, 2], c='r', s=0.5)
        ax.scatter(pts_cpy[:, 0], pts_cpy[:, 1], pts_cpy[:, 2], c='r', s=0.5)
        plt.show()


    print(pts.shape)