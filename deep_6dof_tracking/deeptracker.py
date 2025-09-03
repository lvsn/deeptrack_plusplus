import torch
import math
import time
import numpy as np
import os
import sys

from pytorch_toolbox.io import yaml_load
from deep_6dof_tracking.data.data_augmentation import NormalizeChannels, OffsetDepth
from deep_6dof_tracking.data.deeptrack_bin_loader import DeepTrackBinLoader
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2
from deep_6dof_tracking.data.utils import show_frames, normalize_scale, combine_view_transform, compute_2Dboundingbox
from deep_6dof_tracking.networks.deeptrack_bin_geo_net import DeepTrackBinGeoNet
from deep_6dof_tracking.networks.deeptrack_bin_large_net import DeepTrackBinLargeNet
from deep_6dof_tracking.networks.deeptrack_bin_mask_net import DeepTrackBinMaskNet
from deep_6dof_tracking.networks.deeptrack_bin_net import DeepTrackBinNet
from deep_6dof_tracking.networks.deeptrack_corr_net import DeepTrackCorrNet
from deep_6dof_tracking.networks.deeptrack_dense_net import DeepTrackDenseNet
from deep_6dof_tracking.networks.deeptrack_flow_net import DeepTrackFlowNet
from deep_6dof_tracking.networks.deeptrack_geo_flow_net import DeepTrackGeoFlowNet
from deep_6dof_tracking.networks.deeptrack_geo_net import DeepTrackGeoNet
from deep_6dof_tracking.networks.deeptrack_large2_net import DeepTrackLarge2Net
from deep_6dof_tracking.networks.deeptrack_large_net import DeepTrackLargeNet
from deep_6dof_tracking.networks.deeptrack_large_stride_net import DeepTrackLargeStrideNet
from deep_6dof_tracking.networks.deeptrack_net import DeepTrackNet
from deep_6dof_tracking.networks.deeptrack_contour_net import DeepTrackContourNet
from deep_6dof_tracking.networks.deeptrack_net_se import DeepTrackSENet
from deep_6dof_tracking.networks.deeptrack_net_angle import DeepTrackAngleNet
from deep_6dof_tracking.networks.deeptrack_z_net import DeepTrackZNet

from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.utils.angles import quat2euler

import matplotlib.pyplot as plt


class DeepTracker:
    def __init__(self, camera, backend, architecture="squeeze", debug=False):
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
        self.debug = debug
        # setup model
        if architecture == "squeeze":
            self.tracker_model = DeepTrackNet()
        elif architecture == "squeeze_large":
            self.tracker_model = DeepTrackLargeNet()
        elif architecture == "squeeze_large_stride":
            self.tracker_model = DeepTrackLargeStrideNet()
        elif architecture == "squeeze_z":
            self.tracker_model = DeepTrackZNet()
        elif architecture == "squeeze_large2":
            self.tracker_model = DeepTrackLarge2Net()
        elif architecture == "squeeze_corr":
            self.tracker_model = DeepTrackCorrNet()
        elif architecture == "geo_flow":
            self.tracker_model = DeepTrackGeoFlowNet(show_mask=False)
        elif architecture == "opticalflow":
            self.tracker_model = DeepTrackFlowNet(show_mask=True)
        elif architecture == "mask":
            self.tracker_model = DeepTrackContourNet()
        elif architecture == "angle":
            self.tracker_model = DeepTrackAngleNet()
        elif architecture == "se":
            self.tracker_model = DeepTrackSENet()
        elif architecture == "dense":
            self.tracker_model = DeepTrackDenseNet()
        elif architecture == "geo":
            self.tracker_model = DeepTrackGeoNet()
        elif architecture == "bin":
            self.tracker_model = DeepTrackBinNet(41, log_softmax=False)
        elif architecture == "bin_mask":
            self.tracker_model = DeepTrackBinMaskNet(41, log_softmax=False)
        elif architecture == "bin_cauchy":
            self.tracker_model = DeepTrackBinNet(21, log_softmax=False)
        elif architecture == "bin_large":
            self.tracker_model = DeepTrackBinLargeNet(41, log_softmax=False)
        elif architecture == "bin_geo":
            self.tracker_model = DeepTrackBinGeoNet(41, log_softmax=False, proba=True)
        else:
            print("Error, architecture {} is not supported".format(architecture))
            sys.exit(-1)
        self.cascade_tracker_model = None

        if self.debug and "bin" in architecture:
            plt.ion()
            self.fig = plt.figure(figsize=(15, 12), dpi=80)
            self.axes = []
            self.lines = []
            for i in range(6):
                self.ax = self.fig.add_subplot("61{}".format(i + 1))
                self.axes.append(self.ax)

        self.tracker_model.eval()
        if backend == "cuda":
            self.tracker_model = tracker_model

    def setup_renderer(self, model_3d_path, model_3d_ao_path, shader_path):
        # from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
        #window = InitOpenGL(*self.image_size)

        #self.renderer = ModelRenderer(model_3d_path, shader_path, self.camera, window, self.image_size)
        self.renderer = ModelRenderer2(model_3d_path, shader_path, self.camera, self.image_size)
        if model_3d_ao_path is not None:
            self.renderer.load_ambiant_occlusion_map(model_3d_ao_path)

    def load(self, path, model_3d_path="", model_3d_ao_path="", shader_path="", cascade_path=None):
        self.load_meta_parameters(path)
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.tracker_model.load_state_dict(checkpoint['state_dict'])

        if cascade_path is not None:
            self.load_cascade_net(cascade_path)

        if model_3d_path != "" and model_3d_ao_path != "" and shader_path != "":
            self.setup_renderer(model_3d_path, model_3d_ao_path, shader_path)

    def load_cascade_net(self, cascade_path):
        # cascade tracker
        self.cascade_tracker_model = DeepTrackLargeNet()
        folder_path = os.path.dirname(cascade_path)
        data = yaml_load(os.path.join(folder_path, "meta.yml"))
        self.focus_translation_range = float(data["translation_range"])
        self.focus_rotation_range = float(data["rotation_range"])
        self.focus_object_width = int(data["object_width"]["dragon"])  # TODO should not be a dict
        checkpoint = torch.load(cascade_path, map_location=lambda storage, loc: storage)
        self.cascade_tracker_model.load_state_dict(checkpoint['state_dict'])
        self.cascade_tracker_model.eval()
        if self.backend == "cuda":
            self.cascade_tracker_model = cascade_tracker_model

    def load_meta_parameters(self, path):
        folder_path = os.path.dirname(path)
        data = yaml_load(os.path.join(folder_path, "meta.yml"))
        self.image_size = (int(data["image_size"]), int(data["image_size"]))
        self.translation_range = float(data["translation_range"])
        self.rotation_range = float(data["rotation_range"])

        if "cauchy" in self.architecture:
            self.translation_bins = DeepTrackBinLoader.get_gauss_bins(-self.translation_range, self.translation_range,
                                                                      float(data["translation_variance"]), 21)
            self.rotation_bins = DeepTrackBinLoader.get_gauss_bins(-self.rotation_range, self.rotation_range,
                                                                   float(data["rotation_variance"]), 21)
            self.translation_bins = np.append(-self.translation_range, self.translation_bins)
            self.rotation_bins = np.append(-self.rotation_range, self.rotation_bins)
        else:
            self.translation_bins = DeepTrackBinLoader.get_bins(-self.translation_range, self.translation_range, 41)
            self.rotation_bins = DeepTrackBinLoader.get_bins(-self.rotation_range, self.rotation_range, 41)
        self.translation_bin_center = self.compute_bin_center(self.translation_bins, self.translation_range)
        self.rotation_bin_center = self.compute_bin_center(self.rotation_bins, self.rotation_range)

        if self.debug:
            tags = ["x", "y", "z"]
            for i, ax in enumerate(self.axes[:3]):
                ax.set_ylim([0, 0.5])
                ax.set_xlabel('translation (m)')
                ax.set_title('T {}'.format(tags[i]), rotation='vertical', x=-0.04, y=0.4)
                self.lines.append(ax.bar(np.arange(len(self.translation_bin_center)), np.zeros(self.translation_bins.shape)))
                ax.set_xticks([x for x in np.arange(len(self.translation_bin_center))[::2]])
                ax.set_xticklabels(["{:10.3f}".format(x) for x in self.translation_bin_center[::2]])
            for i, ax in enumerate(self.axes[3:]):
                ax.set_ylim([0, 0.5])
                ax.set_xlabel('rotation (degree)')
                ax.set_title('R {}'.format(tags[i]), rotation='vertical', x=-0.04, y=0.4)
                self.lines.append(ax.bar(np.arange(len(self.rotation_bins)), np.zeros(self.rotation_bins.shape)))
                ax.set_xticks([x for x in np.arange(len(self.rotation_bin_center))[::2]])
                ax.set_xticklabels(["{:10.1f}".format(x) for x in np.degrees(self.rotation_bin_center[::2])])
            plt.tight_layout()
        self.object_width = int(data["object_width"]["dragon"]) #TODO should not be a dict
        self.mean = torch.from_numpy(np.load(os.path.join(folder_path, "mean.npy"))[:, np.newaxis, np.newaxis].astype(np.float32))
        self.std = torch.from_numpy(np.load(os.path.join(folder_path, "std.npy"))[:, np.newaxis, np.newaxis].astype(np.float32))
        if self.backend == "cuda":
            self.mean = mean
            self.std = std
        self.input_bufferA = np.ndarray((1, 4, self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.input_bufferB = np.ndarray((1, 4, self.image_size[0], self.image_size[1]), dtype=np.float32)

    def compute_render(self, pose, bb):
        left = np.min(bb[:, 1])
        right = np.max(bb[:, 1])
        top = np.min(bb[:, 0])
        bottom = np.max(bb[:, 0])
        self.renderer.setup_camera(self.camera, left, right, bottom, top)
        render_rgb, render_depth = self.renderer.render_image(pose)
        return render_rgb, render_depth

    def estimate_current_pose(self, previous_pose, current_rgb, current_depth,
                              debug=False, debug_time=False, raw_prediction=False, cascade=False):
        if debug_time:
            start_time = time.time()
        object_width = self.object_width
        if cascade or self.cascade_tracker_model is not None:
            object_width = self.focus_object_width
        bb = compute_2Dboundingbox(previous_pose, self.camera, object_width, scale=(1000, 1000, -1000))
        bb2 = compute_2Dboundingbox(previous_pose, self.camera, object_width, scale=(1000, -1000, -1000))
        if debug_time:
            print("Compute BB : {}".format(time.time() - start_time))
            start_time = time.time()
        rgbA, depthA = self.compute_render(previous_pose, bb)
        if debug_time:
            print("Render : {}".format(time.time() - start_time))
            start_time = time.time()
        rgbB, depthB = normalize_scale(current_rgb, current_depth, bb2, self.image_size)
        debug_info = (rgbA, bb2, np.hstack((rgbA, rgbB)))
        if debug_time:
            print("Scale : {}".format(time.time() - start_time))
            start_time = time.time()

        rgbA = rgbA.astype(np.float32)
        rgbB = rgbB.astype(np.float32)
        depthA = depthA.astype(np.float32)
        depthB = depthB.astype(np.float32)

        depthA = OffsetDepth.normalize_depth(depthA, previous_pose)
        depthB = OffsetDepth.normalize_depth(depthB, previous_pose)

        if debug:
            show_frames(rgbA, depthA, rgbB, depthB)
        self.input_bufferA[0, 0:3, :, :] = rgbA.T
        self.input_bufferA[0, 3, :, :] = depthA.T
        self.input_bufferB[0, 0:3, :, :] = rgbB.T
        self.input_bufferB[0, 3, :, :] = depthB.T

        imgA = torch
        imgB = torch


        if debug_time:
            print("Load : {}".format(time.time() - start_time))
            start_time = time.time()

        if self.backend == "cuda":
            imgA = imgA.cuda()
            imgB = imgB.cuda()

        imgA[0, :, :, :] -= self.mean[:4, :, :]
        imgA[0, :, :, :] /= self.std[:4, :, :]
        imgB[0, :, :, :] -= self.mean[4:, :, :]
        imgB[0, :, :, :] /= self.std[4:, :, :]

        imgA = torch.autograd.Variable(imgA, requires_grad=True)
        imgB = torch.autograd.Variable(imgB, requires_grad=True)

        if debug_time:
            print("Normalize : {}".format(time.time() - start_time))
            start_time = time.time()

        if not cascade or self.cascade_tracker_model is None:
            prediction = self.tracker_model(imgA, imgB)
        else:
            prediction = self.cascade_tracker_model(imgA, imgB)

        if raw_prediction:
            return prediction, imgA, imgB

        if "bin" in self.architecture:
            if self.debug:
                for i in range(6):
                    distribution = prediction[i].data.cpu().numpy()[0]
                    [rect.set_height(h) for rect, h in zip(self.lines[i], distribution)]
                    [rect.set_color('b') for rect in self.lines[i]]
                    max = np.argmax(distribution)
                    if i < 3:
                        mean = np.digitize(np.sum(self.translation_bins * distribution), self.translation_bins)
                    else:
                        mean = np.digitize(np.sum(self.rotation_bins * distribution), self.rotation_bins)
                    if not "cauchy" in self.architecture:
                        mean -= 1
                    self.lines[i][mean].set_color('r')
                    self.lines[i][max].set_color('g')

                self.fig.canvas.draw()
            if "cauchy" in self.architecture:
                prediction = self.bin_to_pose(prediction, offset=0, method="mean_bin").cpu().numpy()
            else:
                prediction = self.bin_to_pose(prediction, offset=1, method="mean_bin").cpu().numpy()
        else:
            prediction = prediction.data.cpu().numpy()

        if self.architecture == "angle":
            deg_prediction = np.zeros((1, 6))
            deg_prediction[0, :3] = prediction[0, :3]
            deg_prediction[0, 3:] = np.degrees(quat2euler(prediction[0, 3:]))
            prediction = deg_prediction

        if debug_time:
            print("Network time : {}".format(time.time() - start_time))

        if "geo" in self.architecture:
            prediction[:, 3:] = np.degrees(prediction[:, 3:])
        elif self.architecture == "angle":
            prediction[:, :3] *= self.translation_range
        else:
            if cascade and self.cascade_tracker_model is not None:
                prediction = self.unnormalize_label(prediction, self.focus_translation_range, self.focus_rotation_range)
                #prediction[:, 3:] = np.degrees(prediction[:, 3:])
            else:
                #print(self.translation_range, self.rotation_range)
                prediction = self.unnormalize_label(prediction, self.translation_range, self.rotation_range)
        prediction = Transform.from_parameters(*prediction[0], is_degree=True)
        if debug:
            print("Prediction : {}".format(prediction))
        current_pose = combine_view_transform(previous_pose, prediction)
        return current_pose, debug_info

    @staticmethod
    def compute_bin_center(bin, max):
        bin_center = np.zeros(bin.shape)
        for i in range(len(bin)):
            bottom = bin[i]
            if i + 1 == len(bin):
                top = max
            else:
                top = bin[i+1]
            step = top - bottom
            bin_center[i] = bin[i] + step / 2
        return bin_center

    def bin_to_pose(self, predictions, offset=1, method="mean_bin"):
        pose = np.zeros((predictions[0].size(0), 6))
        for i in range(6):
            if method == "max":
                value, argmax = torch.max(predictions[i].data, 1)
                index = argmax.cpu().numpy()
            if i < 3:
                mean = np.sum(self.translation_bin_center * predictions[i].data.cpu().numpy())
                if method == "mean":
                    translation = mean
                elif method == "mean_bin":
                    index = np.digitize(mean, self.translation_bins)
                    translation = self.translation_bin_center[index - offset]
                pose[:, i] = translation
            else:
                mean = np.sum(self.rotation_bin_center * predictions[i].data.cpu().numpy())
                if method == "mean":
                    rotation = mean
                elif method == "mean_bin":
                    index = np.digitize(mean, self.rotation_bins)
                    rotation = self.rotation_bin_center[index - offset]
                pose[:, i] = rotation
        pose = torch.from_numpy(pose).type(predictions[0].data.type())
        pose[:, :3] /= self.translation_range
        pose[:, 3:] /= self.rotation_range

        return pose

    @staticmethod
    def unnormalize_label(params, max_translation, max_rotation_rad):
        params[:, :3] *= max_translation
        params[:, 3:] *= math.degrees(max_rotation_rad)
        return params
