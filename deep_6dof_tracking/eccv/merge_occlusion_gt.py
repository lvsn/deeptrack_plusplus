import argparse
import os
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge video sequences')

    parser.add_argument('--sequences', help="sequence path", action="store")
    parser.add_argument('--output', help="output path", action="store")

    arguments = parser.parse_args()

    sequences = arguments.sequences
    output = arguments.output

    folder = "occlusion"

    ratio = 1
    x_crop_near = (350, 800)
    y_crop_near = (225, 625)
    x_crop_far = (x_crop_near[0], x_crop_near[1])
    y_crop_far = (y_crop_near[0] - 100, y_crop_near[1] - 100)

    sequence_names = ["occlusion_0", "occlusion_v_15", "occlusion_v_30", "occlusion_v_45", "occlusion_v_60"]
    object_names = ["dragon", "shoe", "clock", "skull"]

    output_path = os.path.join(output, folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_video_file = os.path.join(output_path, "occlusion_v.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_video_file, fourcc, 20.0, (150*len(sequence_names), 150*len(object_names)))

    for frame_id in range(150):
        frame = "{}.png".format(frame_id)
        frames = []
        for object in object_names:
            temp_frames= []
            for i, sequence in enumerate(sequence_names):
                x_crop = x_crop_near
                y_crop = y_crop_near
                if "far" in sequence:
                    x_crop = x_crop_far
                    y_crop = y_crop_far
                image = cv2.imread(os.path.join(sequences, "{}_{}".format(object, sequence), frame))
                if image is None:
                    break
                image = image[int(y_crop[0] * ratio):int(y_crop[1] * ratio), int(x_crop[0] * ratio):int(x_crop[1] * ratio), :]
                image = cv2.resize(image, (150, 150))
                temp_frames.append(image)
                if image is None:
                    break
            if image is None:
                break
            frames.append(np.concatenate(temp_frames, axis=1))
        final_frame = np.concatenate(frames, axis=0)
        video.write(final_frame)
        cv2.imshow("test", final_frame)
        cv2.waitKey(1)
    video.release()

