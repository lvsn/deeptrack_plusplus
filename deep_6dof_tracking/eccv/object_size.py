from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.utils.pointcloud import maximum_width
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler
from tqdm import tqdm
import argparse
import shutil
import os
import math
import numpy as np
ESCAPE_KEY = 1048603


if __name__ == '__main__':

    model_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/3D_models"
    models = ["clock", "cookiejar", "dog", "dragon", "lego", "shoe",
              "skull", "walkman", "wateringcan", "turtle", "kinect"]

    print("Compute Max bounding box")
    widths = []
    for model in models:
        geometry_path = os.path.join(model_path, model, "geometry.ply")
        model_3d = PlyParser(geometry_path).get_vertex()
        object_max_width = maximum_width(model_3d) * 1000
        print("{} : {}".format(model, object_max_width))
