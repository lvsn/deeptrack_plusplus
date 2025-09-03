import torch
import math
import time
import numpy as np
import os
import sys

from deep_6dof_tracking.networks.deeptrack_res_net_consistence import DeepTrackConsistence
# attention
# from pytorch_toolbox.io import yaml_load
from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.data_augmentation import NormalizeChannels, OffsetDepth, BoundingBox3D
from deep_6dof_tracking.data.utils import show_frames, normalize_scale, combine_view_transform, compute_2Dboundingbox
from deep_6dof_tracking.networks.deeptrack_res_net import DeepTrackResNet
from deep_6dof_tracking.networks.deeptrack_res_net_grad import DeepTrackResNetGrad
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop import DeepTrackResNetModDrop
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop2 import DeepTrackResNetModDrop2
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop3 import DeepTrackResNetModDrop3
from deep_6dof_tracking.networks.deeptrack_res_net_splitstream import DeepTrackResNetSplit
from deep_6dof_tracking.networks.FPNet import RefineNet

from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.data.model_rend_test import ModelRenderer3

import matplotlib.pyplot as plt
import cv2


class DeepTrackerBatch:
    def __init__(self,
                 camera,
                 backend,
                 architecture="squeeze",
                 deltapose=False,
                 no_token=False,
                 hybrid_vit=False,
                 more_heads=False,
                 smaller=False,
                 bb3d=False,
                 same_mean=False,
                 new_renderer=False):
        self.image_size = None
        self.tracker_model = None
        self.translation_range = None
        self.rotation_range = None
        self.mean = None
        self.std = None
        self.debug_rgb = None
        self.debug_background = None
        self.camera = camera
        self.backend = backend
        self.architecture = architecture
        self.deltapose = deltapose
        self.no_token = no_token
        self.hybrid_vit = hybrid_vit
        self.more_heads = more_heads
        self.smaller = smaller
        self.bb3d = bb3d
        self.same_mean = same_mean
        self.new_renderer = new_renderer

        self.preprocess_total_time = 0
        self.normalize_total_time = 0
        self.network_total_time = 0
        self.n_estimations = 0

        # setup model
        # if it is a class, we assign it directly, if a string we setup the class
        self.tracker_class = architecture
        if type(self.tracker_class) == str:
            if architecture == "res":
                self.tracker_class = DeepTrackResNet
            elif architecture == "res_moddrop":
                self.tracker_class = DeepTrackResNetModDrop
            elif architecture == "res_moddrop2":
                self.tracker_class = DeepTrackResNetModDrop2
            elif architecture == "res_moddrop3":
                self.tracker_class = DeepTrackResNetModDrop3
            elif architecture == "res_grad":
                self.tracker_class = DeepTrackResNetGrad
            elif architecture == "res_consistence":
                self.tracker_class = DeepTrackConsistence
            elif architecture == "res_split":
                self.tracker_class = DeepTrackResNetSplit
            elif architecture == "fpnet":
                self.tracker_class = RefineNet
            else:
                print("Error, architecture {} is not supported".format(architecture))
                sys.exit(-1)

    def setup_renderer(self, model_3d_path, model_3d_ao_path, shader_path, model_scale=1):
        if self.new_renderer:
            print('Using new renderer')
            model_3d_path = os.path.dirname(model_3d_path)
            files = os.listdir(model_3d_path)
            for file in files:
                if file.endswith(".obj"):
                    geometry_path = os.path.join(model_3d_path, file)
                elif file.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    texture_path = os.path.join(model_3d_path, file)
            self.renderer = ModelRenderer3(geometry_path,
                                           shader_path, 
                                           texture_path,
                                           self.camera,
                                           [self.image_size],)
        else:
            self.renderer = ModelRenderer2(model_3d_path, shader_path, self.camera, [self.image_size], backend="glfw",
                                        model_scale=model_scale)
        # self.renderer = ModelRenderer2(model_3d_path, shader_path, self.camera, [self.image_size], backend="glfw",
        #                             model_scale=model_scale)
        if model_3d_ao_path is not None:
            self.renderer.load_ambiant_occlusion_map(model_3d_ao_path)

    def load(self, path, object_name, model_3d_path="", model_3d_ao_path="", shader_path="", object_width=None):
        self.load_meta_parameters(path, object_name)
        if object_width:
            self.bounding_box_width = object_width
        self.load_network(path)

        if model_3d_path != "" and model_3d_ao_path != "" and shader_path != "":
            self.setup_renderer(model_3d_path, model_3d_ao_path, shader_path)
        if self.bb3d == True:
            self.depth_bb = BoundingBox3D(self.bounding_box_width, self.bounding_box, 0.5)
            if self.median_width is not None:
                true_object_width = self.renderer.object_max_width
                bounding_box_width = (self.bounding_box_width * (true_object_width / self.median_width))
                self.depth_bb = BoundingBox3D(bounding_box_width, self.bounding_box, 0.5)

    def load_network(self, path):
        if self.tracker_class in [RefineNet]:
            self.tracker_model = self.tracker_class(cfg={'use_BN': True, 'rot_rep': 'axis_angle'})
        else:
            self.tracker_model = self.tracker_class(image_size=self.image_size[0])#, delta_pose=self.deltapose)
        self.tracker_model.eval()
        if self.backend == "cuda":
            print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Storing ID of current CUDA device
            cuda_id = torch.cuda.current_device()
            print(f"ID of current CUDA device:{torch.cuda.current_device()}")
                
            print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
            self.tracker_model = self.tracker_model.cuda()
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        if self.tracker_class == RefineNet:
            self.tracker_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.tracker_model.load_state_dict(checkpoint['state_dict'])

    def load_meta_parameters(self, path, object_name):
        folder_path = os.path.dirname(path)
        data = yaml_load(os.path.join(folder_path, "meta.yml"))
        self.image_size = (int(data["image_size"]), int(data["image_size"]))
        self.mask = np.ones(self.image_size, dtype=np.float32)
        self.translation_range = float(data["translation_range"])
        self.rotation_range = float(data["rotation_range"])
        try:
            self.bounding_box_width = float(data["object_width"][object_name])
        except KeyError as ke:
            print(f"Error: {ke}")
            self.bounding_box_width = float(data["bounding_box_width"])
        try: 
            self.median_width = float(data["median_width"])
        except KeyError as ke:
            print('No median width specified')
            self.median_width = None
        try:
            self.bounding_box = float(data["bounding_box"])
        except KeyError as ke:
            print('No bounding box specified: Defaulting to 0.15')
            self.bounding_box = 0.15
        self.mean = torch.from_numpy(np.load(os.path.join(folder_path, "mean.npy"))[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32))
        self.std = torch.from_numpy(np.load(os.path.join(folder_path, "std.npy"))[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32))
        if self.backend == "cuda":
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.input_bufferA = np.ndarray((1, 4, self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.input_bufferB = np.ndarray((1, 4, self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.imgA = torch.from_numpy(self.input_bufferA)
        self.imgB = torch.from_numpy(self.input_bufferB)
        # self.imgA.pin_memory()
        # self.imgB.pin_memory()
        if self.backend == "cuda":
            self.imgA = self.imgA.cuda()
            self.imgB = self.imgB.cuda()

    def compute_render(self, pose, bb):
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        self.renderer.setup_camera(self.camera, left, right, bottom, top)
        render_rgb, render_depth = self.renderer.render_image(pose, ambiant_light=np.array([0.5, 0.5, 0.5]),
                                                              light_diffuse=np.array([0.4, 0.4, 0.4]))
        return render_rgb, render_depth

    def estimate_current_pose(self, previous_pose, current_rgb, current_depth,
                              verbose=False, debug_time=False, raw_prediction=False, batch=False,
                              debug_show=False, iteration=0, rgb_only=False):
        
        MY_DEBUG_TIME = False
        intro_time = time.time()

        if debug_time:
            start_time = time.time()
        bounding_box_width = self.bounding_box_width

        noisy_poses = []
        max_r = math.radians(1)
        max_t = 0.0015
        if batch:
            batch_size = len(self.input_bufferA)
            noise_parameters_T_x = [0, 0, max_t, -max_t, 0, max_t, -max_t, max_t, -max_t, 0]
            noise_parameters_T_y = [max_t, -max_t, 0, 0, 0, max_t, -max_t, -max_t, max_t, 2*max_t]
            noise_parameters_T_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        else:
            batch_size = 1
            noise_parameters_T_x = [0]
            noise_parameters_T_y = [0]
            noise_parameters_T_z = [0]

        intro_time = time.time() - intro_time
        if MY_DEBUG_TIME:
            print('intro_time:  ', intro_time)
        render_time = time.time()

        if self.median_width is not None:
            true_object_width = self.renderer.object_max_width
            bounding_box_width = (bounding_box_width * (true_object_width / self.median_width))

        bb = compute_2Dboundingbox(previous_pose, self.camera, bounding_box_width, scale=(1000, 1000, -1000))
        rgbA, depthA = self.compute_render(previous_pose, bb)
        rgbA_debug = rgbA
        rgbA = rgbA.astype(np.float32)
        depthA = depthA.astype(np.float32)
        if iteration == 0:
            self.mask = np.ones(self.image_size, dtype=np.float32)
        rgbA[:, :, :] *= self.mask[:, :, np.newaxis]
        depthA[:, :] *= self.mask
        #import cv2
        #cv2.imshow("eoiw", rgbA.astype(np.uint8))
        depthA = OffsetDepth.normalize_depth(depthA, previous_pose)
        if self.bb3d == True:
            depthA = self.depth_bb.bound_depth(depthA)

        render_time = time.time() - render_time
        if MY_DEBUG_TIME:
            print('render_time:  ', render_time)
        

        for i in range(batch_size):
            pre_time = time.time()
            
            noise = Transform.from_parameters(noise_parameters_T_x[i],
                                              noise_parameters_T_y[i],
                                              noise_parameters_T_z[i],
                                              0, 0, 0)
            noisy_pose = combine_view_transform(previous_pose, noise)
            noisy_poses.append(noisy_pose)
            bb2 = compute_2Dboundingbox(noisy_pose, self.camera, bounding_box_width, scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(current_rgb, current_depth, bb2, self.image_size)
            debug_info = (rgbA_debug, bb2, np.hstack((rgbA_debug, rgbB)))

            rgbB = rgbB.astype(np.float32)
            depthB = depthB.astype(np.float32)
            depthB = OffsetDepth.normalize_depth(depthB, noisy_pose)
            # print(self.bb3d)
            if self.bb3d == True:
                depthB = self.depth_bb.bound_depth(depthB)
                if self.median_width is not None:
                    depthB /= (true_object_width / self.median_width)
                    depthA /= (true_object_width / self.median_width)

            if debug_show:
                vmin = -400
                vmax = 400
                depthA_clipped = np.clip(depthA, vmin, vmax)
                normalized_depthA = ((depthA_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

                depthB_clipped = np.clip(depthB, vmin, vmax)
                normalized_depthB = ((depthB_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

                # normalized_depthA = cv2.applyColorMap(normalized_depthA, cv2.COLORMAP_VIRIDIS)
                # normalized_depthB = cv2.applyColorMap(normalized_depthB, cv2.COLORMAP_VIRIDIS)
                normalized_depthA = cv2.applyColorMap(normalized_depthA, cv2.COLORMAP_JET)
                normalized_depthB = cv2.applyColorMap(normalized_depthB, cv2.COLORMAP_JET)

                rgbA_vis = cv2.resize(rgbA, (0,0), fx=3, fy=3).astype(np.uint8)
                rgbB_vis = cv2.resize(rgbB, (0,0), fx=3, fy=3).astype(np.uint8)
                normalized_depthA = cv2.resize(normalized_depthA, (0,0), fx=3, fy=3).astype(np.uint8)
                normalized_depthB = cv2.resize(normalized_depthB, (0,0), fx=3, fy=3).astype(np.uint8)
                
                row1 = np.hstack((cv2.cvtColor(rgbA_vis, cv2.COLOR_RGB2BGR), cv2.cvtColor(rgbB_vis, cv2.COLOR_RGB2BGR)))
                row2 = np.hstack((normalized_depthA, normalized_depthB))
                full = np.vstack((row1, row2))
                cv2.imshow('debug', full)
                cv2.waitKey(0)
                
            self.imgA[i, 0:3, :, :] = torch.from_numpy(rgbA.T)
            self.imgA[i, 3, :, :] = torch.from_numpy(depthA.T)
            self.imgB[i, 0:3, :, :] = torch.from_numpy(rgbB.T)
            self.imgB[i, 3, :, :] = torch.from_numpy(depthB.T)
            
            pre_time = time.time() - pre_time
            if MY_DEBUG_TIME:
                print('pre_time:  ', pre_time)

        pre_time2 = time.time()

        if debug_time:
            print("Preprocess : {}".format(time.time() - start_time))
            self.preprocess_total_time += time.time() - start_time
            start_time = time.time()

        if self.same_mean == True:
            self.imgA[:, :, :, :] -= self.mean[:, 4:, :, :]
            self.imgA[:, :, :, :] /= self.std[:, 4:, :, :]
        else:
            self.imgA[:, :, :, :] -= self.mean[:, :4, :, :]
            self.imgA[:, :, :, :] /= self.std[:, :4, :, :]
        self.imgB[:, :, :, :] -= self.mean[:, 4:, :, :]
        self.imgB[:, :, :, :] /= self.std[:, 4:, :, :]

        if rgb_only:
            self.imgB[:, 3, :, :] = 0

        with torch.no_grad():
            var_imgA = self.imgA
            var_imgB = self.imgB

            if not batch:
                var_imgA = var_imgA[0, :, :, :].unsqueeze(0)
                var_imgB = var_imgB[0, :, :, :].unsqueeze(0)

        if debug_time:
            print("Normalize : {}".format(time.time() - start_time))
            self.normalize_total_time += time.time() - start_time
            start_time = time.time()

        pre_time2 = time.time() - pre_time2
        if MY_DEBUG_TIME:
            print('pre_time2:  ', pre_time2)
        net_time = time.time()

        prediction, _ = self.tracker_model(var_imgA, var_imgB)

        net_time = time.time() - net_time
        if MY_DEBUG_TIME:
            print('net_time:  ', net_time)
        post_time = time.time()

        if raw_prediction:
            return prediction, var_imgA, var_imgB

        prediction = prediction.data.cpu().numpy()

        if debug_time:
            print("Network time : {}".format(time.time() - start_time))
            self.network_total_time += time.time() - start_time
        self.n_estimations += 1

        prediction = self.unnormalize_label(prediction, self.translation_range, self.rotation_range)

        if self.median_width is not None:
            true_object_width = self.renderer.object_max_width
            prediction[0:3] = prediction[0:3] * true_object_width / self.median_width

        if verbose:
            print("Prediction : {}".format(prediction))
        if batch:
            final_pose = None
            for i, noisy_pose in enumerate(noisy_poses):
                pred = Transform.from_parameters(*prediction[i], is_degree=True)
                current_pose = combine_view_transform(noisy_pose, pred)
                if final_pose is None:
                    final_pose = current_pose.to_parameters()
                else:
                    final_pose += current_pose.to_parameters()
            final_pose = Transform.from_parameters(*(final_pose/len(noisy_poses)))
        else:
            #prediction[0, 2] -= 0.001
            pred_pose = Transform.from_parameters(*prediction[0], is_degree=True)
            final_pose = combine_view_transform(noisy_pose, pred_pose)

        post_time = time.time() - post_time
        if MY_DEBUG_TIME:
            print('post_time:  ', post_time)
            print()

        return final_pose, debug_info
    
    def print_mean_times(self):
        print('-------------------------------------------------')
        print('Mean times')
        print('-------------------------------------------------')
        print("Preprocess : {}".format(round(self.preprocess_total_time / self.n_estimations, 4)))
        print("Normalize : {}".format(round(self.normalize_total_time / self.n_estimations, 4)))
        print("Network : {}".format(round(self.network_total_time / self.n_estimations, 4)))

    @staticmethod
    def unnormalize_label(params, max_translation, max_rotation_rad):
        params[:, :3] *= max_translation
        params[:, 3:] *= math.degrees(max_rotation_rad)
        return params
