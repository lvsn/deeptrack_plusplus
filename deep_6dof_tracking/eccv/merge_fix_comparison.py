import argparse
import os
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge video sequences')

    parser.add_argument('--sequences', help="sequence path", action="store")
    parser.add_argument('--vanilla_sequences', help="sequence path without renders", action="store")
    parser.add_argument('--output', help="output path", action="store")

    arguments = parser.parse_args()

    sequences = arguments.sequences
    vanilla_sequences = arguments.vanilla_sequences
    output = arguments.output

    method_list = [x for x in os.listdir(sequences)]
    sequence_list = [x for x in os.listdir(os.path.join(sequences, method_list[0]))]

    folder = "fix"
    x_crop = (350, 800)
    y_crop = (225, 625)

    output_path = os.path.join(output, folder)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for sequence_name in sequence_list:
        if "fix" in sequence_name:
            if "far" in sequence_name:
                x_crop = (350, 800)
                y_crop = (225, 625)
                x_crop = (350, 800)
                y_crop = (225 - 100, 625 - 100)
            else:
                x_crop = (350, 800)
                y_crop = (225 - 100, 625 - 100)
            output_video_file = os.path.join(output_path, sequence_name + ".avi")
            output_video_file_gt = os.path.join(output_path, sequence_name + "_gt.avi")
            print("Save {}".format(output_video_file))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter(output_video_file, fourcc, 20.0, ((x_crop[1] - x_crop[0]) * 2,
                                                                      (y_crop[1] - y_crop[0]) * 2))
            video_gt = cv2.VideoWriter(output_video_file_gt, fourcc, 20.0, (int((x_crop[1] - x_crop[0])*1.5),
                                                                            int((y_crop[1] - y_crop[0])*1.5)))

            frames = os.listdir(os.path.join(sequences, "single", sequence_name))
            ordered_frames = [int(x[:-4]) for x in frames]
            ordered_frames.sort()
            for frame_id in ordered_frames:
                frame = "{}.png".format(frame_id)
                frame_single = cv2.imread(os.path.join(sequences, "single", sequence_name, frame))
                frame_general = cv2.imread(os.path.join(sequences, "general", sequence_name, frame))
                frame_conv = cv2.imread(os.path.join(sequences, "conv", sequence_name, frame))
                frame_random_forest = cv2.imread(os.path.join(sequences, "random_forest", sequence_name, frame))
                frame_gt = cv2.imread(os.path.join(vanilla_sequences, sequence_name, frame))


                # Crop
                frame_single = frame_single[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1], :]
                frame_general = frame_general[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1], :]
                frame_conv = frame_conv[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1], :]
                frame_random_forest = frame_random_forest[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1], :]
                frame_gt = frame_gt[int(y_crop[0]*1.5):int(y_crop[1]*1.5), int(x_crop[0]*1.5):int(x_crop[1]*1.5), :]

                # Concatenate
                frame_ours = np.concatenate((frame_single, frame_general), axis=1)
                frame_theirs = np.concatenate((frame_conv, frame_random_forest), axis=1)
                frame = np.concatenate((frame_ours, frame_theirs), axis=0)

                #frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
                video.write(frame)
                video_gt.write(frame_gt)
                cv2.imshow("ours", frame)
                cv2.imshow("real", frame_gt)
                cv2.waitKey(20)
            video.release()
            video_gt.release()