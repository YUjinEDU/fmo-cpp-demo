import numpy as np

class ObjectSet:
    def __init__(self):
        self.frames = []
        self.dims = (0, 0)
        self.offset = 0

    def load_ground_truth(self, filename, dims):
        with open(filename, "r") as f:
            lines = f.readlines()
        self.dims = (int(lines[0]), int(lines[1]))
        num_frames = int(lines[2])
        self.offset = int(lines[3])
        num_objects = int(lines[4])
        if self.dims != dims:
            raise ValueError("dimensions inconsistent with video")
        self.frames = [None] * num_frames
        index = 5
        for _ in range(num_objects):
            frame_num = int(lines[index])
            num_runs = int(lines[index + 1])
            index += 2
            if frame_num < 1 or frame_num > num_frames:
                raise ValueError("bad frame number")
            if self.frames[frame_num - 1] is None:
                self.frames[frame_num - 1] = []
            point_set = []
            white = False
            pos = 0
            for _ in range(num_runs):
                run_length = int(lines[index])
                index += 1
                if white:
                    for i in range(pos, pos + run_length):
                        x = i % dims[0]
                        y = i // dims[0]
                        point_set.append((x, y))
                pos += run_length
                white = not white
            if white:
                for i in range(pos, dims[0] * dims[1]):
                    x = i % dims[0]
                    y = i // dims[0]
                    point_set.append((x, y))
            self.frames[frame_num - 1].append(point_set)

    def get(self, frame_num):
        frame_num += self.offset
        if frame_num < 1 or frame_num > len(self.frames):
            return []
        if self.frames[frame_num - 1] is None:
            return []
        return self.frames[frame_num - 1]

    def dims(self):
        return self.dims

    def num_frames(self):
        return len(self.frames)
