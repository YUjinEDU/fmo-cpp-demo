import cv2
import numpy as np
import os
from datetime import datetime

class VideoInput:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video source")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def make_from_camera(cam_id):
        return VideoInput(cam_id)

    @staticmethod
    def make_from_file(filename):
        return VideoInput(filename)

    def receive_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    def set_fps(self, fps_val):
        self.cap.set(cv2.CAP_PROP_FPS, fps_val)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_exposure(self):
        return self.cap.get(cv2.CAP_PROP_EXPOSURE)

    def set_exposure(self, exp_val):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)

    def default_camera(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def dims(self):
        return (self.width, self.height)

    def fps(self):
        return self.fps

class VideoOutput:
    def __init__(self, filename, dims, fps):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.writer = cv2.VideoWriter(filename, fourcc, max(15, fps), (dims[0], dims[1]), True)
        if not self.writer.isOpened():
            raise RuntimeError("Failed to open file for recording")
        self.dims = dims

    @staticmethod
    def make_file(filename, dims, fps):
        return VideoOutput(filename, dims, fps)

    @staticmethod
    def make_in_directory(directory, dims, fps):
        filename = os.path.join(directory, datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi")
        return VideoOutput(filename, dims, fps)

    def send_frame(self, frame):
        if frame.shape[1] != self.dims[0] or frame.shape[0] != self.dims[1]:
            frame = cv2.resize(frame, (self.dims[0], self.dims[1]))
        self.writer.write(frame)
