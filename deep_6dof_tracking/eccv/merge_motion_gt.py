import argparse
import os
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge video sequences')

    parser.add_argument('--sequences', help="sequence path", action="store")
    parser.add_argument('--vanilla_sequences', help="sequence path", action="store")
    parser.add_argument('--output', help="output path", action="store")

    arguments = parser.parse_args()

    sequences = arguments.sequences
    vanilla_sequences = arguments.vanilla_sequences
    output = arguments.output

    folder = "motion"

    ratio = 1.5
    x_crop = (250, 900)
    y_crop = (0, 500)

    sequence_names = ["motion_translation", "motion_rotation", "motion_full", "motion_hard"]
    object_names = ["dragon", "shoe", "clock", "skull"]

    output_path = os.path.join(output, folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_video_file = os.path.join(output_path, "motion.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    for object in object_names:

        for i, sequence in enumerate(sequence_names):
            video = cv2.VideoWriter(os.path.join(output_path, "{}_{}.avi".format(object, sequence)), fourcc, 20.0,
                                    (int((x_crop[1] - x_crop[0]))*2, int((y_crop[1] - y_crop[0]))))

            for frame_id in range(300):
                frame = "{}.png".format(frame_id)
                frames = []
                image = cv2.imread(os.path.join(sequences, "{}_{}".format(object, sequence), frame))
                image_real = cv2.imread(os.path.join(vanilla_sequences, "{}_{}".format(object, sequence), frame))

                if image is None:
                    break
                image = image[int(y_crop[0]):int(y_crop[1]), int(x_crop[0]):int(x_crop[1]), :]
                image_real = image_real[int(y_crop[0] * ratio):int(y_crop[1] * ratio), int(x_crop[0] * ratio):int(x_crop[1] * ratio), :]
                image_real = cv2.resize(image_real, (int((x_crop[1] - x_crop[0])), int((y_crop[1] - y_crop[0]))))
                frame_final = np.concatenate((image_real, image), axis=1)
                video.write(frame_final)
                cv2.imshow("test", frame_final)
                cv2.waitKey(1)
            video.release()

