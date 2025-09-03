from deep_6dof_tracking.utils.transform import Transform
from deep_6dof_tracking.utils.angles import mat2euler
import cv2
import aruco
import numpy as np


class ArucoDetector:
    def __init__(self, camera, board_path):
        self.camparam = aruco.CameraParameters()
        boardconfig = aruco.BoardConfiguration(board_path)
        disto = camera.distortion
        self.camparam.setParams(camera.matrix(),
                                disto,
                                np.array([[camera.width], [camera.height]]))

        # create detector and set parameters
        self.detector = aruco.BoardDetector()
        self.detector.setParams(boardconfig, self.camparam)
        # set minimum marker size for detection
        self.markerdetector = self.detector.getMarkerDetector()
        self.markerdetector.setMinMaxSize(0.01)
        self.likelihood = 1

    def detect(self, img):
        self.likelihood = self.detector.detect_mat(img)
        detection = None
        if self.likelihood > 0.1:
            # get board and draw it
            board = self.detector.getDetectedBoard()
            rvec = board.Rvec.copy()
            tvec = board.Tvec.copy()
            detection = Transform.from_parameters(tvec[0], -tvec[1], -tvec[2], rvec[0][0], -rvec[1][0], -rvec[2][0])
        return detection

    def get_likelihood(self):
        return self.likelihood
