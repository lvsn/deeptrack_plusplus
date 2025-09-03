import math
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count
import argparse
import numpy as np
import cv2
from pyntcloud.samplers import VoxelgridCenters
from pyntcloud.structures import VoxelGrid
from tqdm import tqdm

from torch.utils import data

from pytorch_toolbox.transformations.compose import Compose

from deep_6dof_tracking.data.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, OffsetDepth, NormalizeChannels, ToTensor, ChannelHide, DepthDownsample, KinectOffset
from deep_6dof_tracking.data.deeptrack_loader_base import DeepTrackLoaderBase
from deep_6dof_tracking.data.utils import compute_2Dboundingbox, combine_view_transform

from pyntcloud import PyntCloud
import pandas as pd

from deep_6dof_tracking.utils.transform import Transform


class DeepTrackLoaderTest(DeepTrackLoaderBase):
    """
    Used for occlusion tests
    """
    def __init__(self, root, pretransforms=[], posttransforms=[], target_transform=[]):
        super(DeepTrackLoaderTest, self).__init__(root, pretransforms, posttransforms, target_transform)

    def from_index(self, index):
        rgbA, depthA, initial_pose = self.load_image(index)
        rgbB, depthB, transformed_pose = self.load_pair(index, 0)

        initial_pose = combine_view_transform(initial_pose, transformed_pose)

        sample = [rgbA, depthA, rgbB, depthB, initial_pose.to_parameters()]
        pose_labels = transformed_pose.to_parameters().astype(np.float32)

        pure_depth = depthB.copy()

        pose_labels[:3] /= float(self.metadata["translation_range"])
        pose_labels[3:] /= float(self.metadata["rotation_range"])
        if self.pretransforms:
            sample = self.pretransforms[0](sample)
        mask = sample[-1]
        if mask is None:
            mask = np.ones((150, 150), dtype=np.uint8)
        if self.posttransforms:
            sample = self.posttransforms[0](sample[:-1])
        return [sample, mask, pure_depth], [pose_labels]


def get_3d_model_from_input(pose, depth, camera, bb_width, image_size):
    """
    Compute new camera matrix from the crop/resize process in the dataset generation.
    :param pose:
    :param depth:
    :return:
    """
    new_camera = camera.copy()
    bb = compute_2Dboundingbox(pose, camera, int(float(bb_width)),
                               scale=(1000, -1000, -1000))
    left = np.min(bb[:, 1])
    right = np.max(bb[:, 1])
    top = np.min(bb[:, 0])
    bottom = np.max(bb[:, 0])
    bb_w = right - left
    bb_h = bottom - top
    image_size = int(image_size)
    new_camera.width = image_size
    new_camera.height = image_size
    new_camera.center_x = image_size / 2.
    new_camera.center_y = image_size / 2.
    fov_x = 2 * math.atan2(camera.width, 2 * new_camera.focal_x)
    fov_y = 2 * math.atan2(camera.height, 2 * new_camera.focal_y)
    fov_x = fov_x * bb_w / camera.width
    fov_y = fov_y * bb_h / camera.height
    new_camera.focal_x = new_camera.width / (2 * math.tan(fov_x / 2))
    new_camera.focal_y = new_camera.height / (2 * math.tan(fov_y / 2))

    # back project points
    new_depthA = depth / 1000
    vertex = new_camera.backproject_depth(new_depthA)
    vertex = vertex[vertex.any(axis=1)]  # remove zeros
    vertex[:, 1:] *= -1
    vertex = pose.inverse().dot(vertex)
    return vertex.astype(np.float32)


if __name__ == '__main__':
    #
    #   load configurations from
    #

    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-o', '--output', help="Output path", metavar="FILE", default="./to_delete")
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE", default="/media/ssd/deeptracking/to_delete")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE", default="/home/mathieu/Dataset/RGBD/SUN3D")
    parser.add_argument('-r', '--occluder', help="Occluder path", metavar="FILE",
                        default="/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_official/hand")

    parser.add_argument('-i', '--device', help="Gpu id", action="store", default=0, type=int)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=2, type=int)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)
    parser.add_argument('-g', '--gradientclip', help="Activate gradient clip", action="store_true")


    arguments = parser.parse_args()

    device_id = arguments.device
    backend = arguments.backend
    batch_size = arguments.batchsize
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore
    gradient_clip = arguments.gradientclip

    output_path = arguments.output
    occluder_path = arguments.occluder
    background_path = arguments.background
    data_path = arguments.dataset


    data_path = os.path.expandvars(data_path)
    output_path = os.path.expandvars(output_path)
    occluder_path = os.path.expandvars(occluder_path)
    background_path = os.path.expandvars(background_path)


    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if number_of_core == -1:
        number_of_core = cpu_count()

    loader_class = DeepTrackLoaderTest

    pretransforms = [Compose([Occluder(occluder_path, 0.75)])]

    posttransforms = [Compose([HSVNoise(0.07, 0.05, 0.1),
                               KinectOffset(std=7),
                               Background(background_path),
                               GaussianNoise(2, 20),
                               GaussianBlur(7),
                               DepthDownsample(0.7),
                               OffsetDepth()])]

    print("Load datasets from {}".format(data_path))
    train_dataset = loader_class(os.path.join(data_path, "train"), pretransforms, posttransforms)


    # Instantiate the data loader needed for the train loop. These use dataset object to build random minibatch
    # on multiple cpu core
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=False,
                                   )
    vertex_acc = None
    ratios = []
    stop = 200
    for i, (frame, pose) in tqdm(enumerate(train_loader), total=stop):
        _, _, rgbs, depths, poses = frame[0]
        masks = frame[1]
        pure_depths = frame[2]

        rgbs = rgbs.cpu().numpy()
        depths = depths.cpu().numpy()
        for mask, pure_depth, rgb, depth, pose in zip(masks, pure_depths, rgbs, depths, poses):

            plt.subplot("121")
            plt.imshow(rgb.astype(np.uint8))
            plt.subplot("122")
            plt.imshow(depth)
            plt.show()

            # keep only occluded pixels
            pure_depth = pure_depth.cpu().numpy()
            mask = mask.cpu().numpy()

            # non occluded pixels
            #pure_depth[mask[:, :] == 0] = 0
            # occluded pixels
            occluded_pixels = pure_depth.copy()
            occluded_pixels[mask[:, :] == 1] = 0

            occ_sum = np.sum(occluded_pixels[occluded_pixels != 0])
            total_sum = np.sum(pure_depth[pure_depth != 0])
            #print("occluded : {}".format(occ_sum))
            #print("non occluded : {}".format(total_sum))
            #print("ratio : {}".format(occ_sum/total_sum))
            ratios.append(occ_sum/total_sum)


            # remove borders
            kernel = np.ones((5, 5), np.uint8)
            pure_depth = cv2.morphologyEx(pure_depth, cv2.MORPH_OPEN, kernel)

            """
            plt.subplot("221")
            plt.imshow(pure_depth)
            plt.subplot("222")
            plt.imshow(mask)
            plt.subplot("223")
            plt.imshow(rgb)
            plt.subplot("224")
            plt.imshow(depth)
            plt.show()
            """

            pose = Transform.from_parameters(*pose)

            vertex = get_3d_model_from_input(pose, pure_depth, train_dataset.camera,
                                             train_dataset.metadata["bounding_box_width"],
                                             train_dataset.metadata["image_size"])
            if vertex_acc is None:
                vertex_acc = vertex
            else:
                vertex_acc = np.concatenate((vertex_acc, vertex), axis=0)

            #plt.subplot("221")
            #plt.imshow(rgb.astype(np.uint8))
            #plt.subplot("222")
            #plt.imshow(depth)
            #plt.subplot("223")
            #plt.imshow(mask)
            #plt.subplot("224")
            #plt.imshow(pure_depth)
            #plt.show()
        #break
        if i == stop:
            break

    """
    ratios = np.array(ratios)
    import seaborn as sns
    sns.distplot(ratios)

    points = pd.DataFrame(vertex_acc, columns=['x', 'y', 'z'])

    cloud = PyntCloud(points)

    print("Computing voxel grid...")
    grid = VoxelGrid(cloud, x_y_z=[64, 64, 64])
    grid.extract_info()
    grid.compute()

    grid.plot(mode="density", d=3, cmap="cool")
    plt.show()
    """
