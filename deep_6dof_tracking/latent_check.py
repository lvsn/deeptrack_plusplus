import torch
from deep_6dof_tracking.data.data_augmentation import OffsetDepth
from deep_6dof_tracking.data.rgbd_dataset import RGBDDataset
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.utils.transform import Transform
from pytorch_toolbox.io import yaml_load

from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import combine_view_transform, show_frames, compute_2Dboundingbox, normalize_scale, \
    color_blend
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

    parser = argparse.ArgumentParser(description='Generate synthetic data for DeepTrack')

    parser.add_argument('-c', '--camera', help="camera json path", action="store",
                        default="data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="shader path", action="store", default="data/shaders")
    parser.add_argument('-o', '--output', help="save path", action="store", default="./generated_data")
    parser.add_argument('-m', '--model', help="model file path", action="store", default="./model_config.yaml")

    parser.add_argument('-y', '--saveformat', help="save format", action="store", default="numpy")

    # Data related
    parser.add_argument('-s', '--samples', help="quantity of samples", action="store", default=200000, type=int)
    parser.add_argument('-t', '--translation', help="max translation in meter", action="store", default=0.02, type=float)
    parser.add_argument('-r', '--rotation', help="max rotation in degree", action="store", default=20, type=float)
    parser.add_argument('-a', '--maxradius', help="max distance", action="store", default=1.4, type=float)
    parser.add_argument('-i', '--minradius', help="min distance", action="store", default=0.8, type=float)
    parser.add_argument('--specular', help="Will add random specularity to the training set", action="store_true")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE")
    parser.add_argument('--occluder', help="Occluder path", metavar="FILE")

    parser.add_argument('-v', '--show', help="show image while generating", action="store_true")
    parser.add_argument('-d', '--debug', help="show debug screen while generating", action="store_true")

    # network related
    parser.add_argument('--architecture', help="architecture name", action="store", default="squeeze_large")
    parser.add_argument('--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-n', '--network', help="Network model path", metavar="FILE")



    arguments = parser.parse_args()

    TRANSLATION_RANGE = arguments.translation
    ROTATION_RANGE = math.radians(arguments.rotation)
    SAMPLE_QUANTITY = arguments.samples
    SPHERE_MIN_RADIUS = arguments.minradius
    SPHERE_MAX_RADIUS = arguments.maxradius
    IMAGE_SIZE = (224, 224)
    SAVE_TYPE = arguments.saveformat
    SHOW = arguments.show
    DEBUG = arguments.debug
    SPECULAR = arguments.specular

    SHADER_PATH = arguments.shader
    OUTPUT_PATH = arguments.output
    CAMERA_PATH = arguments.camera
    NETWORK_PATH = arguments.network
    BACKGROUND_PATH = arguments.background
    OCCLUDER_PATH = arguments.occluder

    MODELS = yaml_load(arguments.model)["models"]

    ARCHITECTURE = arguments.architecture
    BACKEND = arguments.backend

    camera = Camera.load_from_json(CAMERA_PATH)
    tracker = DeepTrackerBatch(camera, BACKEND, ARCHITECTURE)
    tracker.load_meta_parameters(NETWORK_PATH, "None")
    tracker.load_network(NETWORK_PATH)

    background_data = RGBDDataset(BACKGROUND_PATH)

    if SHOW:
        import cv2
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    shutil.copy(arguments.model, os.path.join(OUTPUT_PATH, "models.yml"))

    window_size = (camera.width, camera.height)
    sphere_sampler = UniformSphereSampler(SPHERE_MIN_RADIUS, SPHERE_MAX_RADIUS)
    print("Compute Mean bounding box")
    widths = []
    #OBJECT_WIDTH = 220
    latents_rgb = []
    latents_rgbd = []
    predictions = []
    gts = []
    pollos = []
    # Iterate over all models from config files
    for model in MODELS:
        geometry_path = os.path.join(model["path"], "geometry.ply")
        ao_path = os.path.join(model["path"], "ao.ply")
        vpRender = ModelRenderer2(geometry_path, SHADER_PATH, camera, [window_size, IMAGE_SIZE])
        if os.path.exists(ao_path):
            vpRender.load_ambiant_occlusion_map(ao_path)
        for i in tqdm(range(SAMPLE_QUANTITY)):
            random_pose = sphere_sampler.get_random()
            # Sampling from gaussian ditribution in the magnitudes
            random_transform = Transform.from_parameters(0, 0, 0, 0, 0, 0)
            #random_transform = sphere_sampler.random_normal_magnitude(TRANSLATION_RANGE, ROTATION_RANGE)
            pair = combine_view_transform(random_pose, random_transform)
            bb = compute_2Dboundingbox(random_pose, camera, tracker.object_width, scale=(1000, 1000, -1000))
            left = np.min(bb[:, 1])
            right = np.max(bb[:, 1])
            top = np.min(bb[:, 0])
            bottom = np.max(bb[:, 0])
            vpRender.setup_camera(camera, left, right, bottom, top)
            rgbA, depthA = vpRender.render_image(random_pose, fbo_index=1)

            light_intensity = np.zeros(3)
            light_intensity.fill(np.random.uniform(0.1, 1.3))
            light_intensity += np.random.uniform(-0.1, 0.1, 3)
            ambiant_light = np.zeros(3)
            ambiant_light.fill(np.random.uniform(0.5, 0.75))
            shininess = 0
            if np.random.randint(0, 2) and SPECULAR:
                shininess = np.random.uniform(5, 20)
            vpRender.setup_camera(camera, 0, camera.width, camera.height, 0)
            rgbB, depthB = vpRender.render_image(pair,
                                                 fbo_index=0,
                                                 light_direction=sphere_sampler.random_direction(),
                                                 light_diffuse=light_intensity,
                                                 ambiant_light=ambiant_light,
                                                 shininess=shininess)
            rgbB, depthB = normalize_scale(rgbB, depthB, bb, IMAGE_SIZE)

            color_background, depth_background = background_data.load_random_image(rgbB.shape[1])
            depth_background = depth_background.astype(np.int32)
            rgbB, depthB = color_blend(rgbB, depthB, color_background, depth_background)

            depthA = depthA.astype(np.float32)
            depthB = depthB.astype(np.float32)

            depthA = OffsetDepth.normalize_depth(depthA, random_pose)
            depthB = OffsetDepth.normalize_depth(depthB, random_pose)

            tracker.imgA[0, 0:3, :, :] = torch.from_numpy(rgbA.T)
            tracker.imgA[0, 3, :, :] = torch.from_numpy(depthA.T)
            tracker.imgB[0, 0:3, :, :] = torch.from_numpy(rgbB.T)
            tracker.imgB[0, 3, :, :] = torch.from_numpy(depthB.T)
            tracker.imgA[:, :, :, :] -= tracker.mean[:, :4, :, :]
            tracker.imgA[:, :, :, :] /= tracker.std[:, :4, :, :]
            tracker.imgB[:, :, :, :] -= tracker.mean[:, 4:, :, :]
            tracker.imgB[:, :, :, :] /= tracker.std[:, 4:, :, :]

            var_imgA = torch.autograd.Variable(tracker.imgA, volatile=True)
            var_imgB = torch.autograd.Variable(tracker.imgB, volatile=True)

            prediction, latent_rgbd = tracker.tracker_model.stream(var_imgA, var_imgB, None)
            prediction = prediction.data.cpu().numpy()
            prediction = tracker.unnormalize_label(prediction, tracker.translation_range, tracker.rotation_range)[0]
            latent_rgbd = latent_rgbd.data.cpu().numpy()[0]
            rgbd_activation = tracker.tracker_model.probe_activation["moddrop"].detach().cpu().numpy()[0]
            rgbd_means = tracker.tracker_model.probe_activation["means"].detach().cpu().numpy()[0]

            # RGB
            tracker.imgB[0, 3, :, :] = 0
            prediction, latent_rgb = tracker.tracker_model.stream(var_imgA, var_imgB, None)
            prediction = prediction.data.cpu().numpy()
            prediction = tracker.unnormalize_label(prediction, tracker.translation_range, tracker.rotation_range)[0]
            latent_rgb = latent_rgb.data.cpu().numpy()[0]
            rgb_activation = tracker.tracker_model.probe_activation["moddrop"].detach().cpu().numpy()[0]
            rgb_means = tracker.tracker_model.probe_activation["means"].detach().cpu().numpy()[0]

            # Show activation
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.subplot(1, 2, 1)
            sns.distplot(rgbd_means)
            sns.distplot(rgb_means)
            plt.subplot(1, 2, 2)
            sns.distplot(np.abs(rgb_means - rgbd_means))
            plt.show()

            activation_error = np.abs(rgbd_activation - rgb_activation)
            #mean_channels = list(np.mean(activation_error, axis=(1, 2)))
            #pollos += mean_channels
            #sns.distplot(pollos, norm_hist=True)
            #plt.show()
            #"""
            filters = tracker.tracker_model.fireB.squeeze.weight.data.detach().cpu().numpy()
            filters = np.abs(filters)
            c = 20
            print("RGB filter sum : {}".format(np.sum(filters[:, :32, :, :])))
            print("D filter sum : {}".format(np.sum(filters[:, 32:, :, :])))
            #count = 1
            #for j in range(c):
            #    plt.subplot(c, 2, count)
            #    plt.imshow(np.sum(filters[j, 0:3, :, :], axis=1), vmin=0, vmax=0.5)
            #    plt.subplot(c, 2, count+1)
            #    plt.imshow(filters[j, 3, :, :], vmin=0, vmax=0.5)
            #    count += 2
            #plt.show()


            # Here we print a few activations
            c = 7
            count = 1
            for j in range(c):
                minimum = np.min(rgbd_activation[j, :, :])
                maximum = np.max(rgbd_activation[j, :, :])
                plt.subplot(c, 3, count)
                plt.imshow(rgb_activation[j, :, :], vmin=minimum, vmax=maximum)
                plt.title("RGB")
                plt.subplot(c, 3, count+1)
                plt.imshow(rgbd_activation[j, :, :], vmin=minimum, vmax=maximum)
                plt.title("RGBD")
                plt.subplot(c, 3, count+2)
                plt.imshow(activation_error[j, :, :], vmin=0, vmax=1)
                plt.title("Abs Error")
                count += 3
            plt.show()
            #"""
            latents_rgb.append(latent_rgb)
            latents_rgbd.append(latent_rgbd)
            predictions.append(prediction)
            gts.append(random_transform.to_parameters())

            if DEBUG:
                show_frames(rgbA, depthA, rgbB, depthB)
            if SHOW:
                cv2.imshow("test", np.concatenate((rgbA[:, :, ::-1], rgbB[:, :, ::-1]), axis=1))
                k = cv2.waitKey(1)
                if k == ESCAPE_KEY:
                    break

    latents_rgb = np.array(latents_rgb)
    latents_rgbd = np.array(latents_rgbd)
    predictions = np.array(predictions)
    gts = np.array(gts)

    np.save(os.path.join(OUTPUT_PATH, "latent_rgb.npy"), latents_rgb)
    np.save(os.path.join(OUTPUT_PATH, "latent_rgbd.npy"), latents_rgbd)
    np.save(os.path.join(OUTPUT_PATH, "predictions.npy"), predictions)
    np.save(os.path.join(OUTPUT_PATH, "gts.npy"), gts)

