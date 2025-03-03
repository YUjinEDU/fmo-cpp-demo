import os
import threading
import cv2
import numpy as np

class RecordingThread:
    def __init__(self, dir, format, dims, fps):
        self.format = format
        self.dims = dims
        self.video_output = VideoOutput.make_in_directory(dir, dims, fps)
        self.stop = False
        self.exchange = Exchange(format, dims)
        self.thread = threading.Thread(target=self.thread_impl)
        self.thread.start()

    def thread_impl(self):
        input_image = np.zeros((self.dims[1], self.dims[0], 3), dtype=np.uint8)
        while not self.stop:
            self.exchange.swap_receive(input_image)
            if self.stop:
                return
            self.video_output.send_frame(input_image)

    def swap_send(self, input_image):
        self.exchange.swap_send(input_image)

    def __del__(self):
        self.stop = True
        self.exchange.exit()
        self.thread.join()

class AutomaticRecorder:
    NUM_FRAMES = 60

    def __init__(self, dir, format, dims, fps):
        self.dir = dir
        self.format = format
        self.dims = dims
        self.fps = fps
        self.images = [np.zeros((dims[1], dims[0], 3), dtype=np.uint8) for _ in range(self.NUM_FRAMES)]
        self.head = 0
        self.stop_at = 0
        self.thread = None
        self.frame_num = 0

    def frame(self, input_image, event):
        self.frame_num += 1
        if self.thread and self.stop_at == self.head:
            self.thread = None
        if event:
            if not self.thread:
                self.thread = RecordingThread(self.dir, self.format, self.dims, self.fps)
            self.stop_at = self.head
        self.head = (self.head + 1) % self.NUM_FRAMES
        if self.thread and self.frame_num > self.NUM_FRAMES:
            self.thread.swap_send(self.images[self.head])
        self.images[self.head] = input_image.copy()

    def is_recording(self):
        return self.thread is not None

    def __del__(self):
        while self.thread:
            self.frame(self.images[self.head], False)

class ManualRecorder:
    def __init__(self, dir, format, dims, fps):
        self.image = np.zeros((dims[1], dims[0], 3), dtype=np.uint8)
        self.thread = RecordingThread(dir, format, dims, fps)

    def frame(self, input_image):
        self.image = input_image.copy()
        self.thread.swap_send(self.image)

    def __del__(self):
        pass

class VideoOutput:
    @staticmethod
    def make_in_directory(dir, dims, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = os.path.join(dir, f"output_{int(time.time())}.avi")
        return cv2.VideoWriter(filename, fourcc, fps, (dims[0], dims[1]))

    def send_frame(self, frame):
        self.write(frame)

class Exchange:
    def __init__(self, format, dims):
        self.format = format
        self.dims = dims
        self.buffer = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def swap_send(self, input_image):
        with self.lock:
            self.buffer = input_image.copy()
            self.condition.notify()

    def swap_receive(self, output_image):
        with self.lock:
            while self.buffer is None:
                self.condition.wait()
            output_image[:] = self.buffer
            self.buffer = None

    def exit(self):
        with self.lock:
            self.buffer = None
            self.condition.notify_all()
