import vispy.scene
from vispy.scene import visuals
from vispy.io import read_mesh
from vispy.io.wavefront import WavefrontReader
from vispy import app, gloo
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from deep_6dof_tracking.data.utils import compute_2Dboundingbox
from deep_6dof_tracking.utils.plyparser import PlyParser

from PIL import Image
import OpenGL.GL as gl

import os
import numpy as np
import trimesh
#from trimesh import Geometry

from deep_6dof_tracking.utils.pointcloud import maximum_width


class ModelRenderer3(app.Canvas):
    def __init__(self, model_path, shader_path, texture_path, camera, window_sizes, original_width=None, new_width=None, mesh_center=None):
        app.Canvas.__init__(self, show=False, size=(camera.width, camera.height))

        self.window_sizes = window_sizes
        self.camera = camera
        self.rgb = np.array([])
        self.depth = np.array([])

        fragment_code = open(os.path.join(shader_path, "fragment_light3.txt"), 'r').read()
        vertex_code = open(os.path.join(shader_path, "vertex_light3.txt"), 'r').read()

        # Load the OBJ file
        # vertices, faces, normals, texcoords = read_mesh(model_path)
        mesh = trimesh.load(model_path, force='mesh')

        if mesh_center is not None:
            mesh.vertices -= mesh_center

        vertices, faces, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
        try:
            texcoords = mesh.visual.uv
        except AttributeError:
            texcoords = np.zeros((len(mesh.vertices), 2))

        self.max_width_original = maximum_width(vertices)*1000
        print(f'Original width: {self.max_width_original}')
        print(f'Maximum vertices width: {vertices.max(axis=0)}')
        print(f'Minimum vertices width: {vertices.min(axis=0)}')

        if new_width is not None:
            print(f'ratio : {new_width / self.max_width_original}')
            vertices = vertices * (new_width / self.max_width_original)

        self.max_width = maximum_width(vertices)*1000
        self.object_max_width = self.max_width

        self.data = np.ones(vertices.shape[0], [('a_position', np.float32, 3),
                                           ('a_color', np.float32, 3),
                                           ('a_normal', np.float32, 3),
                                           ('a_ambiant_occlusion', np.float32, 3),
                                           ('a_texcoords', np.float32, 2)])
        
        self.data['a_position'] = vertices
        self.data['a_normal'] = normals / np.linalg.norm(self.data['a_normal'], axis=1)[:, np.newaxis]
        self.faces = faces


        # set white texture by default
        texture = np.ones((1, 1, 4), dtype=np.uint8)
        texture.fill(255)
        try:
            self.data['a_texcoords'] = texcoords
            texture = np.array(Image.open(texture_path)).astype(np.uint8)
            if texture is not None:
                texture = texture[::-1, :, :]
        except KeyError:
            pass

        # Calculate face areas
        # face_areas = []
        # for face in self.faces:
        #     v1, v2, v3 = self.data['a_position'][face]
        #     edge1 = v2 - v1
        #     edge2 = v3 - v1
        #     area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        #     face_areas.append(area)
        # # Total surface area
        # total_area = sum(face_areas)
        # self.face_areas = np.array(face_areas) / total_area

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

    def setup_camera(self, camera, left, right, bottom, top):
        self.near_plane = 0.01
        self.far_plane = 5

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

if __name__ == '__main__':
    
    model_path = 'E:/omniobject3d/downsampled/toy_truck_009/Scan.obj'
    texture_path = 'E:/omniobject3d/downsampled/toy_truck_009/Scan.jpg'


    shader_path = 'deep_6dof_tracking/data/shaders'
    camera_path = 'deep_6dof_tracking/data/sensors/camera_parameter_files/synthetic.json'

    camera = Camera.load_from_json(camera_path)
    window_size = (camera.width, camera.height)
    renderer = ModelRenderer3(model_path, shader_path, texture_path, camera, [window_size, (174, 174)])

    sphere_sampler = UniformSphereSampler(0.8, 1.4)
    #sphere_sampler = UniformSphereSampler(10, 10)
    random_pose = sphere_sampler.get_random()
    light_intensity = np.zeros(3)
    light_intensity.fill(np.random.uniform(0.1, 1.3))
    light_intensity += np.random.uniform(-0.1, 0.1, 3)
    ambiant_light = np.zeros(3)
    ambiant_light.fill(np.random.uniform(0.5, 0.75))
    shininess = 0
    # if np.random.randint(0, 2):
    #     shininess = np.random.uniform(3, 30)
    print(renderer.max_width)
    with_add = 15 / 100 * renderer.max_width
    max_width = int(renderer.max_width + with_add)
    print(max_width)
    bb = compute_2Dboundingbox(random_pose, camera, max_width, scale=(1000, 1000, -1000))
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])
    renderer.setup_camera(camera, left, right, bottom, top)
    rgbB, depthB = renderer.render_image(random_pose,
                                            fbo_index=1,
                                            light_direction=sphere_sampler.random_direction(),
                                            light_diffuse=light_intensity,
                                            ambiant_light=ambiant_light,
                                            shininess=shininess)


    import cv2
    import matplotlib.pyplot as plt
    plt.subplot(121)
    rgbB = rgbB[:, :, ::-1]
    rgbB = cv2.cvtColor(rgbB, cv2.COLOR_BGR2RGB)
    plt.imshow(rgbB)
    plt.subplot(122)
    normalized_depthB = (depthB - np.min(depthB)) / (np.max(depthB) - np.min(depthB))
    plt.imshow(normalized_depthB)
    plt.show()