import cv2
import matplotlib.pyplot as plt
import torch
from deep_6dof_tracking.data.data_augmentation import OffsetDepth

from deep_6dof_tracking.data.utils import normalize_scale, combine_view_transform
from deep_6dof_tracking.deeptracker_batch import DeepTrackerBatch
from deep_6dof_tracking.utils.camera import Camera
import os
import numpy as np
from tqdm import tqdm

from deep_6dof_tracking.utils.transform import Transform

object_sizes = {"01": 150,
                "02": 230}

if __name__ == '__main__':

    DEBUG = False
    scene_id = "02"

    ARCHITECTURE = "res_consistence"
    NETWORK_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/models/tracking/generic_da4_best_iccv/model_last.pth.tar"
    OUTPUT_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/ssd6d_refine/{}".format(scene_id)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    MODEL_3D_PATH = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/detection/test_dataset/hinterstoisser/models"
    SHADER_PATH = "../data/shaders"
    BACKEND = "cuda"

    ssd6d_sequence_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/Results/ssd6d/{}".format(scene_id)
    ids = list(set([int(x.split("_")[0]) for x in os.listdir(ssd6d_sequence_path)]))
    ids.sort()

    img_path = "/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/dataset/detection/test_dataset/hinterstoisser/test/{}/rgb".format(scene_id)

    # orginal camera center : centers=(325.26110, 242.04899)  (wtf!)
    camera = Camera(focal=(572.41140, 573.57043), centers=(327.26110, 242.04899), size=(640, 480))
    tracker = DeepTrackerBatch(camera, BACKEND, ARCHITECTURE)
    tracker.load_meta_parameters(NETWORK_PATH, "None")
    tracker.load_network(NETWORK_PATH)
    tracker.setup_renderer(os.path.join(MODEL_3D_PATH, "obj_{}.ply".format(scene_id)),
                           None,
                           SHADER_PATH, model_scale=0.001)
    flipyz = Transform.scale(1, -1, -1)

    for id in tqdm(ids):
        gt_pose = np.load(os.path.join(ssd6d_sequence_path, "{:04}_gt.npy".format(id)))
        est_pose = np.load(os.path.join(ssd6d_sequence_path, "{:04}_est.npy".format(id)))

        img = cv2.imread(os.path.join(img_path, "{:04}.png".format(id)))[:, :, ::-1]

        gt_T = Transform.from_matrix(gt_pose)
        est_T = Transform.from_matrix(est_pose)
        est_T = flipyz.combine(est_T, copy=True)

        if DEBUG:
            gt_T = flipyz.combine(gt_T, copy=True)
            bb = tracker.compute_2Dboundingbox(gt_T, camera, object_sizes[scene_id], scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(gt_T, bb)

            bb2 = tracker.compute_2Dboundingbox(gt_T, camera, object_sizes[scene_id], scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(img, np.zeros_like(img)[:, :, 0], bb2, (174, 174))

            plt.subplot(3, 4, 1)
            plt.imshow(img)
            plt.subplot(3, 4, 2)
            plt.imshow(rgbA)
            plt.subplot(3, 4, 3)
            plt.imshow(rgbB)
            plt.subplot(3, 4, 4)
            rgbB[rgbA != 0] = 0
            plt.imshow(rgbB)

            bb = tracker.compute_2Dboundingbox(est_T, camera, object_sizes[scene_id], scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(est_T, bb)

            bb2 = tracker.compute_2Dboundingbox(est_T, camera, object_sizes[scene_id], scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(img, np.zeros_like(img)[:, :, 0], bb2, (174, 174))

            plt.subplot(3, 4, 5)
            plt.imshow(img)
            plt.subplot(3, 4, 6)
            plt.imshow(rgbA)
            plt.subplot(3, 4, 7)
            plt.imshow(rgbB)
            plt.subplot(3, 4, 8)
            rgbB_copy = rgbB.copy()
            rgbB_copy[rgbA != 0] = 0
            plt.imshow(rgbB_copy)

        final_pose = est_T.copy()
        for i in range(5):
            bb = tracker.compute_2Dboundingbox(final_pose, camera, object_sizes[scene_id], scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(final_pose, bb)

            bb2 = tracker.compute_2Dboundingbox(final_pose, camera, object_sizes[scene_id], scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(img, np.zeros_like(img)[:, :, 0], bb2, (174, 174))

            depthA = depthA.astype(np.float32)
            depthB = depthB.astype(np.float32)

            depthA = OffsetDepth.normalize_depth(depthA, final_pose)
            depthB = OffsetDepth.normalize_depth(depthB, final_pose)

            tracker.imgA[0, 0:3, :, :] = torch.from_numpy(rgbA.T)
            tracker.imgA[0, 3, :, :] = torch.from_numpy(depthA.T)
            tracker.imgB[0, 0:3, :, :] = torch.from_numpy(rgbB.T)
            tracker.imgB[0, 3, :, :] = torch.from_numpy(depthB.T)
            tracker.imgA[:, :, :, :] -= tracker.mean[:, :4, :, :]
            tracker.imgA[:, :, :, :] /= tracker.std[:, :4, :, :]
            tracker.imgB[:, :, :, :] -= tracker.mean[:, 4:, :, :]
            tracker.imgB[:, :, :, :] /= tracker.std[:, 4:, :, :]

            tracker.imgB[0, 3, :, :] = 0

            var_imgA = torch.autograd.Variable(tracker.imgA, volatile=True)
            var_imgB = torch.autograd.Variable(tracker.imgB, volatile=True)

            prediction, latent_rgbd = tracker.tracker_model.stream(var_imgA, var_imgB, None)
            prediction = prediction.data.cpu().numpy()
            prediction = tracker.unnormalize_label(prediction, tracker.translation_range, tracker.rotation_range)[0]
            pred_pose = Transform.from_parameters(*prediction, is_degree=True)
            final_pose = combine_view_transform(final_pose, pred_pose)

        final_pose_flip = flipyz.inverse().combine(final_pose, copy=True)
        np.save(os.path.join(OUTPUT_PATH, "{:04}_est.npy".format(id)), final_pose_flip.matrix)
        np.save(os.path.join(OUTPUT_PATH, "{:04}_gt.npy".format(id)), gt_pose)

        if DEBUG:
            #print(pred_pose)
            #print(final_pose.to_parameters())
            #print(est_T.to_parameters())

            bb = tracker.compute_2Dboundingbox(final_pose, camera, object_sizes[scene_id], scale=(1000, 1000, -1000))
            rgbA, depthA = tracker.compute_render(final_pose, bb)

            bb2 = tracker.compute_2Dboundingbox(final_pose, camera, object_sizes[scene_id], scale=(1000, -1000, -1000))
            rgbB, depthB = normalize_scale(img, np.zeros_like(img)[:, :, 0], bb2, (174, 174))

            plt.subplot(3, 4, 9)
            plt.imshow(img)
            plt.subplot(3, 4, 10)
            plt.imshow(rgbA)
            plt.subplot(3, 4, 11)
            plt.imshow(rgbB)
            plt.subplot(3, 4, 12)
            rgbB[rgbA != 0] = 0
            plt.imshow(rgbB)

            plt.show()
