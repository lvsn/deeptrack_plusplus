import abc
from enum import Enum

from multiprocessing import Queue, Process
import cv2

ESCAPE_KEY = 1048603
SPACE_KEY = 1048608

class ImageShowMessage(Enum):
    Quit = 0
    Space = 1

class ImageShow:
    def __init__(self, show_zoom, path, fps, width, height, encoding="XVID", max_buffer_size=-1):
        self.show_zoom = show_zoom
        self.messages = Queue()
        self.frames = Queue()
        self.fps = fps
        self.path = path
        self.encoding = encoding
        self.width = width
        self.height = height
        self.encoder = None
        self.recording = False

    def show_frame(self, frame):
        self.frames.put(frame)

    def get_message(self):
        msg = None
        if self.messages.qsize() != 0:
            msg = self.messages.get(block=True, timeout=None)
        return msg

    def __enter__(self):
        self.worker = Process(target=self.saver_, args=(self.frames,))
        self.worker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.frames.put(None)
        self.worker.join()

    def saver_(self, frames):
        fourcc = cv2.VideoWriter_fourcc(*self.encoding)
        self.encoder = cv2.VideoWriter(self.path, fourcc, self.fps,
                                       (int(self.width/2), int(self.height/2)))

        while True:
            frame = frames.get(block=True, timeout=None)
            if frame is None:
                break
            self.save_frame_(frame)
        self.encoder.release()
        print("Out of encoder")

    @abc.abstractmethod
    def save_frame_(self, frame):
        """
        Receive a frame and execute save code
        :return:
        """
        self.encoder.write(frame[:, :, ::-1])
