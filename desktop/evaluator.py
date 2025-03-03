import os
import numpy as np
import cv2

class Event:
    TP = 0
    TN = 1
    FP = 2
    FN = 3

class Comparison:
    NONE = 0
    SAME = 1
    IMPROVEMENT = 2
    REGRESSION = 3
    BUFFERING = 4

def event_name(event):
    if event == Event.TP:
        return "TP"
    elif event == Event.TN:
        return "TN"
    elif event == Event.FP:
        return "FP"
    elif event == Event.FN:
        return "FN"
    else:
        return "??"

class EvalResult:
    def __init__(self):
        self.eval = {Event.TP: 0, Event.TN: 0, Event.FP: 0, Event.FN: 0}
        self.comp = Comparison.NONE
        self.iou_dt = []
        self.iou_gt = []

    def clear(self):
        self.eval = {Event.TP: 0, Event.TN: 0, Event.FP: 0, Event.FN: 0}
        self.comp = Comparison.NONE
        self.iou_dt = []
        self.iou_gt = []

    def __str__(self):
        result = ""
        for event in [Event.FN, Event.FP, Event.TN, Event.TP]:
            if self.eval[event] > 0:
                if self.eval[event] > 1:
                    result += f"{self.eval[event]}x"
                result += f"{event_name(event)} "
        if self.comp == Comparison.IMPROVEMENT:
            result += "(improvement) "
        elif self.comp == Comparison.REGRESSION:
            result += "(regression) "
        elif self.comp == Comparison.SAME:
            result += "(no change) "
        elif self.comp == Comparison.NONE:
            result += "(no baseline) "
        elif self.comp == Comparison.BUFFERING:
            result += "(buffering)"
        return result

def good(eval):
    return eval[Event.FN] + eval[Event.FP] == 0

def bad(eval):
    return eval[Event.TN] + eval[Event.TP] == 0

class FileResults:
    IOU_STORAGE_FACTOR = 1e3

    def __init__(self, name):
        self.name = name
        self.frames = []
        self.iou = []

    def clear(self):
        self.frames = []
        self.iou = []

class Results:
    def __init__(self):
        self.list = []
        self.map = {}

    def new_file(self, name):
        if name in self.map:
            self.map[name].clear()
            return self.map[name]
        else:
            file_results = FileResults(name)
            self.list.append(file_results)
            self.map[name] = file_results
            return file_results

    def get_file(self, name):
        if name in self.map:
            return self.map[name]
        else:
            return FileResults("(no results)")

    def load(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
        intro_token = "/FMO/EVALUATION/V3/"
        if intro_token not in lines[0]:
            raise ValueError("failed to find data start token")
        num_files = int(lines[1])
        index = 2
        for _ in range(num_files):
            name = lines[index].strip()
            file_results = self.new_file(name)
            num_frames = int(lines[index + 1])
            num_ious = int(lines[index + 2])
            file_results.frames = [{} for _ in range(num_frames)]
            index += 3
            for event in [Event.FN, Event.FP, Event.TN, Event.TP]:
                event_name_str = event_name(event)
                if event_name_str not in lines[index]:
                    raise ValueError(f"expected {event_name_str} but got {lines[index]}")
                index += 1
                for frame in file_results.frames:
                    frame[event] = int(lines[index])
                    index += 1
            if num_ious > 0:
                file_results.iou = [int(lines[index + i]) for i in range(num_ious)]
                index += num_ious
            if index >= len(lines):
                raise ValueError("error while parsing")

    def save(self, file):
        with open(file, "w") as f:
            f.write("/FMO/EVALUATION/V3/\n")
            f.write(f"{len(self.map)}\n")
            for name, file_results in self.map.items():
                f.write(f"{name} {len(file_results.frames)} {len(file_results.iou)}\n")
                for event in [Event.FN, Event.FP, Event.TN, Event.TP]:
                    f.write(event_name(event))
                    for frame in file_results.frames:
                        f.write(f" {frame[event]}")
                    f.write("\n")
                if len(file_results.iou) > 0:
                    f.write("IOU")
                    for value in file_results.iou:
                        f.write(f" {value}")
                    f.write("\n")

    def make_iou_histogram(self, bins):
        divisor = int(round(FileResults.IOU_STORAGE_FACTOR / bins))
        hist = [0] * bins
        for file_results in self.list:
            for value in file_results.iou:
                bin_index = min(bins - 1, value // divisor)
                hist[bin_index] += 1
        return hist

    def get_average_iou(self):
        total_sum = 0
        count = 0
        for file_results in self.list:
            for value in file_results.iou:
                total_sum += value
                count += 1
        if count == 0:
            return 0.0
        return total_sum / (count * FileResults.IOU_STORAGE_FACTOR)

class Evaluator:
    FRAME_OFFSET = -1

    def __init__(self, gt_filename, dims, results, baseline):
        self.gt = ObjectSet()
        self.gt.load_ground_truth(gt_filename, dims)
        self.name = extract_sequence_name(gt_filename)
        self.file = results.new_file(self.name)
        self.file.frames = [{} for _ in range(self.gt.num_frames())]
        self.baseline = baseline.get_file(self.name)
        if len(self.baseline.frames) == 0:
            self.baseline = None
        if self.baseline and len(self.baseline.frames) != self.gt.num_frames():
            raise ValueError("bad baseline number of frames")
        self.frame_num = 0

    def evaluate_frame(self, dt, frame_num, out, iou_threshold):
        if self.frame_num + 1 != frame_num:
            raise ValueError("bad number of frames")
        self.frame_num = frame_num
        if self.frame_num > self.gt.num_frames():
            raise ValueError("movie length inconsistent with GT")
        gt = self.gt.get(self.frame_num)
        out.clear()
        out.iou_dt = [0.0] * len(dt.detections)
        out.iou_gt = [0.0] * len(gt)
        for i, dt_detection in enumerate(dt.detections):
            dt_points = dt_detection.get_points()
            for j, gt_points in enumerate(gt):
                score = iou(dt_points, gt_points)
                out.iou_gt[j] = max(out.iou_gt[j], score)
                out.iou_dt[i] = max(out.iou_dt[i], score)
        for score in out.iou_gt:
            if score > iou_threshold:
                out.eval[Event.TP] += 1
            else:
                out.eval[Event.FN] += 1
        for score in out.iou_dt:
            if score > iou_threshold:
                pass
            else:
                out.eval[Event.FP] += 1
        for score in out.iou_gt:
            if score > 0:
                self.file.iou.append(int(round(score * FileResults.IOU_STORAGE_FACTOR)))
        if len(dt.detections) == 0 and len(gt) == 0:
            out.eval[Event.TN] += 1
        if self.baseline:
            baseline = self.baseline.frames[self.file.frames.index(out.eval)]
            if bad(baseline) and good(out.eval):
                out.comp = Comparison.IMPROVEMENT
            elif good(baseline) and bad(out.eval):
                out.comp = Comparison.REGRESSION
            else:
                out.comp = Comparison.SAME
        else:
            out.comp = Comparison.NONE
        self.file.frames[self.file.frames.index(out.eval)] = out.eval

def extract_filename(path):
    return os.path.basename(path)

def extract_sequence_name(path):
    filename = extract_filename(path)
    for suffix in [".mat", ".txt", "_gt", ".avi", ".mp4", ".mov"]:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    return filename.replace(" ", "_")

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

    def num_frames(self):
        return len(self.frames)

def iou(ps1, ps2):
    intersection = 0
    union = 0
    for p1 in ps1:
        if p1 in ps2:
            intersection += 1
        union += 1
    for p2 in ps2:
        if p2 not in ps1:
            union += 1
    return intersection / union
