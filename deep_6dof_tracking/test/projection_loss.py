from torch.autograd import Variable
import numpy as np

from deep_6dof_tracking.data.modelrenderer import ModelRenderer, InitOpenGL
from deep_6dof_tracking.data.utils import combine_view_transform, compute_2Dboundingbox, normalize_scale
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.utils.plyparser import PlyParser
from deep_6dof_tracking.networks.rodrigues_function import RodriguesFunction, RodriguesModule
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn
from deep_6dof_tracking.utils.uniform_sphere_sampler import UniformSphereSampler


def hook_a(grad_output):
    print(grad_output)

PROJECTION = True
ZOOM_PROJECTION = False
SHOW = True
DEPTH_POINT = False


def coord_to_image(camera, coords):
    img = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    pixels = coords.data.cpu().numpy()[0].astype(int).T
    pixels[:, 0] = np.clip(pixels[:, 0], 0, camera.width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, camera.height - 1)
    img[pixels[:, 1], pixels[:, 0], 0] = 255
    return img

if __name__ == '__main__':
    model_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_official/3dmodels/dragon/geometry.ply"
    ao_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/deeptracking_official/3dmodels/dragon/ao.ply"
    output_path = "/home/mathieu/Downloads/test/"
    camera_path = "/home/mathieu/source/deep_6dof_tracking/deep_6dof_tracking/data/sensors/camera_parameter_files/synthetic.json"
    shader_path = "../data/shaders"

    camera = Camera.load_from_json(camera_path)
    camera_copy = camera.copy()

    window_size = (camera.width, camera.height)
    window = InitOpenGL(*window_size)
    vpRender = ModelRenderer(model_path, shader_path, camera, window, window_size)
    vpRender.load_ambiant_occlusion_map(ao_path)

    model = PlyParser(model_path).get_vertex().T
    initial_pose = np.array([0, 0, -0.75, 0, -80, -145], dtype=np.float32)
    transform = Transform.from_parameters(*initial_pose, is_degree=True)

    steps = 10
    steps_tx = np.linspace(-0.00, 0.00, steps)
    steps_ty = np.linspace(-0.02, 0.02, steps)
    #steps_ty = np.linspace(0, 0, steps)
    steps_tz = np.linspace(0.00, -0.00, steps)
    #steps_tz = np.linspace(0, 0, steps)
    steps_rx = np.linspace(math.radians(-30), math.radians(30), steps)
    #steps_rx = np.linspace(0, 0, steps)
    steps_ry = np.linspace(math.radians(-15), math.radians(15), steps)
    #steps_ry = np.linspace(0, 0, steps)
    steps_rz = np.linspace(math.radians(10), math.radians(-10), steps)
    #steps_rz = np.linspace(0, 0, steps)

    transform_log = np.zeros((6, steps))
    grad_log = np.zeros((6, steps))
    count = 0

    for tx, ty, tz, rx, ry, rz in tqdm(zip(steps_tx, steps_ty, steps_tz, steps_rx, steps_ry, steps_rz)):
        #sphere_sampler = UniformSphereSampler(0.4, 1.2)
        #initial_pose = sphere_sampler.get_random().to_parameters()
        #transform = Transform.from_parameters(*initial_pose)

        delta_pose = np.array([tx, ty, tz, rx, ry, rz])
        #delta_pose = Transform.random((-0.02, 0.02), (-30, 30)).to_parameters()

        transform_log[:, count] = delta_pose

        delta_transform = Transform.from_parameters(*delta_pose)
        pose = transform.to_parameters().astype(np.float32)
        delta_pose = delta_transform.to_parameters().astype(np.float32)

        torch_model = torch.from_numpy(model).unsqueeze(0)
        torch_angles = torch.from_numpy(pose[3:].copy()).unsqueeze(0)
        torch_translation = torch.from_numpy(pose[:3].copy()).unsqueeze(0)

        torch_delta_angles = torch.from_numpy(delta_pose[3:].copy()).unsqueeze(0)
        torch_delta_translation = torch.from_numpy(delta_pose[:3].copy()).unsqueeze(0)

        torch_model = Variable(torch_model, requires_grad=True)
        torch_angles = Variable(torch_angles, requires_grad=True)
        torch_translation = Variable(torch_translation, requires_grad=True)
        torch_delta_angles = Variable(torch_delta_angles, requires_grad=True)
        torch_delta_translation = Variable(torch_delta_translation, requires_grad=True)

        # Compute
        R = RodriguesModule()(torch_angles).view(-1, 3, 3)
        T = torch_translation
        delta_R = RodriguesModule()(torch_delta_angles).view(-1, 3, 3)
        new_R = torch.bmm(delta_R, R)
        new_T = torch_translation + torch_delta_translation

        #PlyParser.save_points(pointcloud_R_gt.data[0].cpu().numpy().T, os.path.join(output_path, "gt.ply"))
        #PlyParser.save_points(pointcloud_R.data[0].cpu().numpy().T,
        #                      os.path.join(output_path, "prediction.ply"))


        rgbA, depthA = vpRender.render_image(transform)
        transformed_pose = combine_view_transform(transform, delta_transform)
        rgbB, depthB = vpRender.render_image(transformed_pose)
        bb = compute_2Dboundingbox(transform, camera, 250, scale=(1000, -1000, -1000))
        rgbA, depthA = normalize_scale(rgbA, depthA, bb, (150, 150))
        rgbB, depthB = normalize_scale(rgbB, depthB, bb, (150, 150))

        # Apply zoom around model
        if ZOOM_PROJECTION:
            left = np.min(bb[:, 1])
            right = np.max(bb[:, 1])
            top = np.min(bb[:, 0])
            bottom = np.max(bb[:, 0])
            bb_w = right - left
            bb_h = bottom - top
            camera_copy.width = 150
            camera_copy.height = 150
            camera_copy.center_x = 150./2.
            camera_copy.center_y = 150./2.
            fov_x = 2 * math.atan2(camera.width, 2*camera.focal_x)
            fov_y = 2 * math.atan2(camera.height, 2*camera.focal_y)
            fov_x = fov_x * bb_w / camera.width
            fov_y = fov_y * bb_h / camera.height
            camera_copy.focal_x = camera_copy.width / (2*math.tan(fov_x/2))
            camera_copy.focal_y = camera_copy.height / (2*math.tan(fov_y/2))

        # Back project Z from image A and use it as model
        if DEPTH_POINT:
            new_depthA = depthA/1000
            new_model = camera_copy.backproject_depth(new_depthA)
            new_model = new_model[new_model.any(axis=1)] #remove zeros
            new_model[:, 1:] *= -1
            new_model = transform.inverse().dot(new_model)
            new_model = new_model.astype(np.float32).T
            import os
            PlyParser.save_points(new_model.T, os.path.join(output_path, "gt.ply"))
            PlyParser.save_points(torch_model.data[0].cpu().numpy().T, os.path.join(output_path, "prediction.ply"))
            #print(torch_model)
            torch_model = Variable(torch.from_numpy(new_model).unsqueeze(0), requires_grad=True)
            #print(torch_model)

        # Setup camera matrix
        K = torch.from_numpy(camera_copy.matrix().astype(np.float32)).unsqueeze(0)
        K[0, 0, 0] *= -1
        K = Variable(K, requires_grad=True)

        pointcloud_R_gt = T.unsqueeze(-1) + torch.bmm(R, torch_model)
        pointcloud_R = T.unsqueeze(-1) + torch.bmm(new_R, torch_model)

        pointcloud_T_gt = T.unsqueeze(-1) + torch.bmm(R, torch_model)
        pointcloud_T = new_T.unsqueeze(-1) + torch.bmm(R, torch_model)

        norm_n = 2
        if PROJECTION:
            pointcloud_R = torch.bmm(K, pointcloud_R)
            pointcloud_T = torch.bmm(K, pointcloud_T)
            pointcloud_R_gt = torch.bmm(K, pointcloud_R_gt)
            pointcloud_T_gt = torch.bmm(K, pointcloud_T_gt)

            projected_R_gt_model = pointcloud_R_gt[:, :, :] / pointcloud_R_gt[:, 2, :]
            projected_R_prediction_model = pointcloud_R[:, :, :] / pointcloud_R[:, 2, :]
            projected_T_gt_model = pointcloud_T_gt[:, :, :] / pointcloud_T_gt[:, 2, :]
            projected_T_prediction_model = pointcloud_T[:, :, :] / pointcloud_T[:, 2, :]

            if SHOW:
                R_prediction = coord_to_image(camera_copy, projected_R_prediction_model)
                R_gt = coord_to_image(camera_copy, projected_R_gt_model)
                plt.imshow(R_gt[:, :, 0], cmap="gray")
                plt.show()
                R_prediction[R_gt[:, :, 0] == 255, :] = [0, 255, 0]
                T_prediction = coord_to_image(camera_copy, projected_T_prediction_model)
                T_gt = coord_to_image(camera_copy, projected_T_gt_model)
                T_prediction[T_gt[:, :, 0] == 255, :] = [0, 255, 0]
                #cv2.imwrite("gif/{:04}.png".format(count), np.concatenate((R_prediction, T_prediction), axis=1))
                plt.subplot("221")
                plt.imshow(rgbA)
                plt.subplot("222")
                plt.imshow(rgbB)
                plt.subplot("223")
                plt.imshow(R_prediction)
                plt.subplot("224")
                plt.imshow(T_prediction)
                plt.show()

            mse_R = torch.sum((projected_R_prediction_model[:, :2, :] - projected_R_gt_model[:, :2, :]) ** norm_n, 1) / 2
            mse_T = torch.sum((projected_T_prediction_model[:, :2, :] - projected_T_gt_model[:, :2, :]) ** norm_n, 1) / 2
            loss = mse_R.mean() + mse_T.mean()
        else:
            #print((prediction_model - gt_model).pow(2).mean(2))
            mse_R = torch.sum((pointcloud_R - pointcloud_R_gt)**norm_n, 1)/pointcloud_R.size(1)
            mse_T = torch.sum((pointcloud_T - pointcloud_T_gt)**norm_n, 1)/pointcloud_T.size(1)
            loss = mse_R.mean() + mse_T.mean()

        loss.backward()
        grad_log[:3, count] = torch_delta_translation.grad.data.numpy()
        grad_log[3:, count] = torch_delta_angles.grad.data.numpy()
        count += 1

    X = np.arange(steps)
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots(2, 2)

    # Translation
    for label, log in zip(labels, transform_log[:3]):
        ax[0, 0].plot(X, log, label=label)
    for label, log in zip(labels, grad_log[:3]):
        ax[0, 1].plot(X, log, label=label)

    # Rotation
    for label, log in zip(labels, transform_log[3:]):
        ax[1, 0].plot(X, log, label=label)
    for label, log in zip(labels, grad_log[3:]):
        ax[1, 1].plot(X, log, label=label)

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    #ax[0, 1].set_ylim([-0.01, 0.01])
    #ax[1, 1].set_ylim([-0.01, 0.01])
    plt.show()
