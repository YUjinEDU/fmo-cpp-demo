import time
import cv2
import numpy as np
from evaluator import Evaluator, EvalResult, Event, Comparison
from video import VideoInput
from recorder import ManualRecorder, DetectionReport
from objectset import ObjectSet
from algorithm import Algorithm

class Statistics:
    def __init__(self):
        self.total_detections = 0
        self.n_frames = 0

    def next_frame(self, n_detections):
        self.total_detections += n_detections
        self.n_frames += 1

    def get_mean(self):
        return self.total_detections / self.n_frames

    def print(self):
        print(f"Detections: total - {self.total_detections}, average - {self.get_mean()}")

def process_video(status, input_num):
    if len(status.args.names) > input_num:
        print(f"Processing {status.args.names[input_num]}")
    input = VideoInput.make_from_file(status.args.inputs[input_num]) if not status.have_camera() else VideoInput.make_from_camera(status.args.camera)

    if status.args.exposure != 100:
        input.set_exposure(status.args.exposure)
    if status.args.fps != -1:
        input.set_fps(status.args.fps)
    print(f"Exposure value: {input.get_exposure()}")
    print(f"FPS: {input.fps()}")

    dims = input.dims()
    fps = input.fps()

    status.input_name = status.args.inputs[input_num] if not status.have_camera() else f"camera {status.args.camera}"

    evaluator = None
    if status.args.gts:
        evaluator = Evaluator(status.args.gts[input_num], dims, status.results, status.baseline)
    elif status.args.gt_dir:
        gt_path = f"{status.args.gt_dir}{status.args.names[input_num]}.txt"
        evaluator = Evaluator(gt_path, dims, status.results, status.baseline)

    sequence_report = None
    if status.rpt:
        sequence_report = status.rpt.make_sequence(status.input_name)

    if not status.have_camera():
        wait_sec = status.args.wait / 1e3 if status.have_wait() else 1 / fps
        status.window.set_frame_time(wait_sec)

    format = status.args.yuv if status.args.yuv else "BGR"
    object_vec = [None]
    algorithm = Algorithm.make(status.args.params, format, dims)
    frame = None
    frame_copy = np.zeros((dims[1], dims[0], 3), dtype=np.uint8)
    output_cache = None
    eval_result = EvalResult()
    status.in_frame_num = 1
    status.out_frame_num = 1 + algorithm.get_output_offset()

    stat = Statistics()
    while not status.quit and not status.reload:
        status.in_frame_num += 1
        status.out_frame_num += 1

        allow_new_frames = True
        if evaluator:
            num_gt_frames = evaluator.gt().num_frames()
            if status.out_frame_num > num_gt_frames:
                break
            elif status.in_frame_num > num_gt_frames:
                allow_new_frames = False

        if allow_new_frames:
            frame = input.receive_frame()
            if frame is None:
                break
            if status.have_camera():
                frame = cv2.flip(frame, 1)

        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) if format == "YUV" else frame.copy()
        algorithm.set_input_swap(frame_copy)
        output_cache = algorithm.get_output(False)
        stat.next_frame(len(output_cache.detections))

        if evaluator:
            if status.out_frame_num >= 1:
                evaluator.evaluate_frame(output_cache, status.out_frame_num, eval_result, status.args.params.iou_threshold)
                if status.args.pause_fn and eval_result.eval[Event.FN] > 0:
                    status.paused = True
                if status.args.pause_fp and eval_result.eval[Event.FP] > 0:
                    status.paused = True
                if status.args.pause_rg and eval_result.comp == Comparison.REGRESSION:
                    status.paused = True
                if status.args.pause_im and eval_result.comp == Comparison.IMPROVEMENT:
                    status.paused = True
            else:
                eval_result.clear()
                eval_result.comp = Comparison.BUFFERING

        if sequence_report:
            sequence_report.write_frame(status.out_frame_num, output_cache, eval_result)

        if status.args.frame == status.in_frame_num:
            status.unset_frame()
            status.paused = True

        if status.have_frame():
            continue

        if status.args.headless and not status.paused:
            continue

        status.visualizer.visualize(status, frame, evaluator, eval_result, algorithm)

    stat.print()
    input.default_camera()
    return stat
