from deep_6dof_tracking.data.sequence_loader import Sequence
import cv2
import numpy as np


if __name__ == '__main__':
    sequence_path = 'sequence_gen/festo_cam2'
    sequence_path = 'simkinect'
    out_video_path = f'{sequence_path}/video.mp4'
    out_video_path_depth = f'{sequence_path}/video_d.mp4'
    img_size = (640, 480)

    DEPTHONLY = True
    video_data = Sequence(sequence_path, depthonly=DEPTHONLY, preload=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, 15.0, img_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_depth = cv2.VideoWriter(out_video_path_depth, fourcc, 15.0, img_size)

    for i in range(video_data.size()):
        current_rgb, current_depth = video_data.load_image(i)
        
        heatmap = cv2.convertScaleAbs(current_depth, alpha=255/2400)
        heatmap = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
        if DEPTHONLY:
            current_rgb = np.zeros_like(heatmap)
        
        out_video.write(current_rgb)
        out_video_depth.write(heatmap)

        cv2.imshow('rgb', np.concatenate((current_rgb, heatmap), axis=1))
        cv2.waitKey(1)

    if not DEPTHONLY:
        out_video.release()
    out_video_depth.release()