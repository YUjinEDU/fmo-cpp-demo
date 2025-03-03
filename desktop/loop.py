import argparse
import os
import time
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from threading import Thread

class Status:
    def __init__(self, args):
        self.args = args
        self.paused = False
        self.quit = False
        self.reload = False
        self.sound = False
        self.input_string = "Baseline"
        self.in_frame_num = 1
        self.out_frame_num = 1
        self.results = []
        self.baseline = []
        self.date = datetime.now()
        self.timer = time.time()
        self.input_name = ""
        self.visualizer = None
        self.rpt = None

    def have_camera(self):
        return self.args.camera != -1

    def have_wait(self):
        return self.args.wait != -1

    def have_frame(self):
        return self.args.frame != -1

    def unset_frame(self):
        self.args.frame = -1

class Visualizer:
    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        pass

class DebugVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.level = 1
        self.show_lm = False
        self.show_im = True
        self.add = 0
        self.vis = None
        self.output_cache = None
        self.object_points = []
        self.points_cache = None
        self.gt_points_cache = None
        self.detections = 0
        self.pause_first = False
        self.previous_det = 0

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.process(status, frame, evaluator, eval_result, algorithm)
        cv2.imshow("Debug Visualizer", self.vis)
        self.process_keyboard(status, frame)

    def process(self, status, frame, evaluator, eval_result, algorithm):
        self.stats.append(time.time())
        fps_last = len(self.stats) / (self.stats[-1] - self.stats[0])

        self.vis = algorithm.get_debug_image(self.level, self.show_im, self.show_lm, self.add)
        self.output_cache = algorithm.get_output(False)
        self.object_points = [detection.get_points() for detection in self.output_cache.detections]
        self.detections = len(self.output_cache.detections)

        if evaluator:
            status.window.print(eval_result.str())
            gt = evaluator.gt().get(status.out_frame_num)
            self.points_cache = self.merge_point_sets(self.object_points)
            self.gt_points_cache = self.merge_point_sets(gt)
            self.draw_points_gt(self.points_cache, self.gt_points_cache, self.vis)
            status.window.set_text_color(self.good(eval_result.eval))
        else:
            self.draw_points(self.points_cache, self.vis, (255, 0, 255))

    def process_keyboard(self, status, frame):
        step = False
        while status.paused and not step and not status.quit:
            command = status.window.get_command(status.paused)
            if command == "PAUSE":
                status.paused = not status.paused
            if command == "STEP":
                step = True
            if command == "QUIT":
                status.quit = True
            if command == "SCREENSHOT":
                cv2.imwrite("screenshot.png", self.vis)
            if command == "LEVEL0":
                self.level = 0
            if command == "LEVEL1":
                self.level = 1
            if command == "LEVEL2":
                self.level = 2
            if command == "LEVEL3":
                self.level = 3
            if command == "LEVEL4":
                self.level = 4
            if command == "LEVEL5":
                self.level = 5
            if command == "SHOW_IM":
                self.show_im = not self.show_im
            if command == "LOCAL_MAXIMA":
                self.show_lm = not self.show_lm
            if command == "SHOW_NONE":
                self.add = 0
            if command == "DIFF":
                self.add = 1
            if command == "BIN_DIFF":
                self.add = 2
            if command == "DIST_TRAN":
                self.add = 3

            if not status.have_camera():
                if command == "JUMP_BACKWARD":
                    status.paused = False
                    status.args.frame = max(1, status.in_frame_num - 10)
                    status.reload = True
                if command == "JUMP_FORWARD":
                    status.paused = False
                    status.args.frame = status.in_frame_num + 10

    def merge_point_sets(self, point_sets):
        merged = []
        for point_set in point_sets:
            merged.extend(point_set)
        return merged

    def draw_points_gt(self, points, gt_points, vis):
        for point in points:
            cv2.circle(vis, (point[0], point[1]), 2, (0, 255, 0), -1)
        for gt_point in gt_points:
            cv2.circle(vis, (gt_point[0], gt_point[1]), 2, (0, 0, 255), -1)

    def draw_points(self, points, vis, color):
        for point in points:
            cv2.circle(vis, (point[0], point[1]), 2, color, -1)

    def good(self, eval):
        return eval[0] == 0 and eval[2] == 0

class DemoVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.show_help = False
        self.automatic = None
        self.manual = None
        self.record_annotations = True
        self.segments = []
        self.curves = []
        self.events_detected = 0
        self.max_detections = 0
        self.last_detect_frame = -12
        self.forced_event = False
        self.output = None
        self.vis = None
        self.number_detections = 0
        self.offset_from_max_detection = 0
        self.speeds = []
        self.last_speeds = []
        self.max_speed = 0
        self.mode = 0

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.process(status, frame, algorithm)
        cv2.imshow("Demo Visualizer", self.vis)
        self.process_keyboard(status, frame)

    def process(self, status, frame, algorithm):
        self.stats.append(time.time())
        fps_estimate = len(self.stats) / (self.stats[-1] - self.stats[0])

        self.output = algorithm.get_output(True)
        self.number_detections = len(self.output.detections)

        if self.number_detections > self.max_detections:
            self.max_detections = self.number_detections
            self.offset_from_max_detection = 0
        else:
            self.offset_from_max_detection += 1

        if self.offset_from_max_detection > 300 and self.number_detections > 0:
            self.offset_from_max_detection = 0
            self.max_detections = self.number_detections

        self.vis = frame.copy()

        for detection in self.output.detections:
            self.on_detection(status, detection)

        self.draw_segments(self.vis)
        self.print_status(status, fps_estimate)

    def process_keyboard(self, status, frame):
        if self.automatic:
            event = self.forced_event or len(self.output.detections) > 0
            if self.record_annotations:
                self.automatic.frame(self.vis, event)
            else:
                self.automatic.frame(frame, event)
        elif self.manual:
            if self.record_annotations:
                self.manual.frame(self.vis)
            else:
                self.manual.frame(frame)
        self.forced_event = False

        step = False
        while status.paused and not status.quit and not step:
            command = status.window.get_command(False)
            if command == "PAUSE":
                status.paused = not status.paused
            if command == "INPUT":
                self.handle_input(status)
            if command == "STEP":
                step = True
            if command == "QUIT":
                status.quit = True
                self.manual = None
            if command == "SHOW_HELP":
                self.show_help = not self.show_help
                self.update_help(status)
            if command == "AUTOMATIC_MODE":
                if self.manual:
                    self.manual = None
                if not self.automatic:
                    self.automatic = AutomaticRecorder(status.args.record_dir, frame.format, frame.shape[:2], 30)
                    self.update_help(status)
            if command == "FORCED_EVENT":
                self.forced_event = True
            if command == "MANUAL_MODE":
                if self.automatic:
                    self.automatic = None
                    self.update_help(status)
            if command == "RECORD_GRAPHICS":
                self.record_annotations = not self.record_annotations
            if command == "RECORD":
                if self.manual:
                    self.manual = None
                elif not self.automatic:
                    self.manual = ManualRecorder(status.args.record_dir, frame.format, frame.shape[:2], 30)
            if command == "PLAY_SOUNDS":
                status.sound = not status.sound
            if command == "LEVEL0":
                self.mode = 0
            if command == "LEVEL1":
                self.mode = 1
            if command == "LEVEL2":
                self.mode = 2
            if command == "LEVEL3":
                self.mode = 3
            if command == "LEVEL4":
                self.mode = 4

    def handle_input(self, status):
        n_players = 10
        string_size = 10
        vis_table = status.window.vis_table
        status.window.vis_table = False
        sp_now = round(self.max_speed * 100) / 100

        if len(status.window.table) >= n_players:
            last_el = status.window.table[-1]
            if sp_now <= last_el[0]:
                status.window.set_center_line("Sorry, not in top " + str(n_players), "")
                cv2.imshow("Demo Visualizer", self.vis)
                cv2.waitKey(500)
                status.window.vis_table = vis_table
                status.window.set_center_line("", "")
                cv2.imshow("Demo Visualizer", self.vis)
                return

        status.window.set_center_line("Input player's name", "")
        cv2.imshow("Demo Visualizer", self.vis)

        key_code = ' '
        vec = []
        while key_code != 13 and key_code != '\n' and key_code != 10:
            if key_code != ' ':
                if key_code == 127 or key_code == 8:
                    if len(vec) > 0:
                        vec.pop()
                else:
                    vec.append(key_code)
                str_input = ''.join(vec)
                status.window.set_center_line("Input player's name", str_input)
                cv2.imshow("Demo Visualizer", self.vis)
            key_code = cv2.waitKey(0)

        status.window.set_center_line("", "")
        cv2.imshow("Demo Visualizer", self.vis)
        str_input = ''.join(vec)
        status.input_string = str_input

        player_name = status.input_string
        if len(player_name) < string_size:
            player_name = player_name + ' ' * (string_size - len(player_name))
        if len(player_name) > string_size:
            player_name = player_name[:string_size]

        if len(status.window.table) < n_players:
            status.window.table.append((sp_now, player_name))
        else:
            last_el = status.window.table[-1]
            if sp_now > last_el[0]:
                status.window.table[-1] = (sp_now, player_name)
        status.window.table.sort(reverse=True)
        status.window.vis_table = vis_table

    def update_help(self, status):
        if not self.show_help:
            status.window.set_bottom_line("")
        else:
            help_text = "[esc] quit"
            if self.automatic:
                help_text += " | [m] manual mode | [e] forced event"
            else:
                help_text += " | [a] automatic mode | [r] start/stop recording"
            if status.sound:
                help_text += " | [s] disable sound"
            else:
                help_text += " | [s] enable sound"
            status.window.set_bottom_line(help_text)

    def print_status(self, status, fps_estimate):
        recording = self.automatic.is_recording() if self.automatic else bool(self.manual)
        kmh = True
        meas = " km/h" if kmh else " mph"
        fctr = 1 if kmh else 0.621371

        status.window.print("Detections: " + str(self.max_detections))
        for i, speed in enumerate(self.speeds):
            status.window.print("Speed " + str(i + 1) + " : " + str(round(speed * fctr * 100) / 100)[:4] + meas)
        status.window.print("Max speed: " + str(round(self.max_speed * fctr * 100) / 100)[:4] + meas, (255, 0, 0))

    def on_detection(self, status, detection):
        if status.out_frame_num - self.last_detect_frame > 12:
            if status.sound:
                print('\a')
            self.events_detected += 1
            self.segments.clear()
            self.curves.clear()
        self.last_detect_frame = status.out_frame_num

        if detection.predecessor.have_center():
            segment = (detection.predecessor.center, detection.object.center)
            self.segments.append(segment)
        elif detection.object.curve:
            curve = detection.object.curve.clone()
            curve.scale = detection.object.scale
            self.curves.append(curve)
            radius_cm = status.args.radius
            sp = 0
            fps_real = 29.97
            if status.args.fps != -1:
                fps_real = status.args.fps

            if status.args.p2cm == -1:
                sp = detection.object.velocity * 3600 * fps_real * radius_cm * 1e-5
            else:
                length = detection.object.velocity * (detection.object.radius + 1.5)
                sp = length * status.args.p2cm * fps_real * 3600 * 1e-5
            self.speeds.append(sp)

            if len(self.last_speeds) > 20:
                self.last_speeds[19] = (60, sp)
            else:
                self.last_speeds.append((60, sp))

            self.last_speeds.sort(reverse=True)

        speed_now = 0
        for i, (time_left, speed) in enumerate(self.last_speeds):
            if time_left > 0:
                self.last_speeds[i] = (time_left - 1, speed)
                if speed > speed_now:
                    speed_now = speed

        if detection.object.curve:
            self.max_speed = speed_now

        if len(self.segments) > 200:
            self.segments = self.segments[len(self.segments) // 2:]

        if len(self.curves) > 20:
            self.curves = self.curves[-20:]

    def draw_segments(self, vis):
        color = (255, 0, 255)
        thickness = 8
        for segment in self.segments:
            color = (max(color[0], color[0] + 2), max(color[1], color[1] + 1), max(color[2], color[2] + 4))
            cv_color = (color[0], color[1], color[2])
            pt1 = (segment[0][0], segment[0][1])
            pt2 = (segment[1][0], segment[1][1])
            cv2.line(vis, pt1, pt2, cv_color, thickness)
        for curve in self.curves:
            color = (max(color[0], color[0] + 2), max(color[1], color[1] + 1), max(color[2], color[2] + 4))
            cv_color = (color[0], color[1], color[2])
            curve.draw_smooth(vis, cv_color, thickness)

class TUTDemoVisualizer(Visualizer):
    def __init__(self, status):
        self.vis1 = DemoVisualizer(status)
        self.vis1.mode = 2
        self.input = []
        self.vis = None
        self.temp_debug = None
        self.temp2_debug = None
        self.offset_from_max = 0
        self.last_detected_image = None
        self.max_detected_image = None
        self.last_mode = -1

        try:
            self.input.append(VideoInput.make_from_file("../data/webcam/circle_back_res.avi"))
            self.input.append(VideoInput.make_from_file("../data/webcam/counting_all_res.avi"))
            self.input.append(VideoInput.make_from_file("../data/webcam/floorball_res.avi"))
        except Exception as e:
            self.input.clear()
            try:
                self.input.append(VideoInput.make_from_file("data/webcam/circle_back_res.avi"))
                self.input.append(VideoInput.make_from_file("data/webcam/counting_all_res.avi"))
                self.input.append(VideoInput.make_from_file("data/webcam/floorball_res.avi"))
            except Exception as e:
                pass

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        if not self.vis1.manual and status.args.no_record:
            self.vis1.manual = ManualRecorder(status.args.record_dir, frame.format, frame.shape[:2], 30)
        self.vis1.process(status, frame, algorithm)
        if self.last_detected_image is None:
            self.last_detected_image = frame.copy()
        if self.max_detected_image is None:
            self.max_detected_image = frame.copy()
        if self.vis1.previous_detections * self.vis1.number_detections > 0:
            self.last_detected_image = self.vis1.vis.copy()
        if self.vis1.offset_from_max_detection == 0:
            self.max_detected_image = self.vis1.vis.copy()
            self.offset_from_max = 0
        else:
            self.offset_from_max += 1

        if self.vis1.mode == 0:
            self.vis = np.zeros((2 * frame.shape[0], 2 * frame.shape[1], 3), dtype=np.uint8)
            imgs = []
            for in_video in self.input:
                next_frame = in_video.receive_frame()
                if next_frame is None:
                    in_video.restart()
                    next_frame = in_video.receive_frame()
                imgs.append(next_frame)
            for i in range(len(imgs), 4):  # Fill remaining slots with the current visualization
                imgs.append(self.vis1.vis)
            self.vis = self.imgridfull(imgs, 2, 2)
            cv2.imshow("TUT Demo Visualizer", self.vis)
        elif self.vis1.mode == 1:
            cv2.imshow("TUT Demo Visualizer", self.vis1.vis)
        elif self.vis1.mode == 2:
            self.vis = np.zeros((2 * frame.shape[0], 2 * frame.shape[1], 3), dtype=np.uint8)
            imgs = [frame]
            self.temp_debug = algorithm.get_debug_image(1, True, False, 1)
            self.temp_debug = cv2.resize(self.temp_debug, (frame.shape[1], frame.shape[0]))
            imgs.append(self.temp_debug)
            self.temp2_debug = algorithm.get_debug_image(1, True, True, 3)
            self.temp2_debug = cv2.resize(self.temp2_debug, (frame.shape[1], frame.shape[0]))
            imgs.append(self.temp2_debug)
            imgs.append(self.vis1.vis)
            self.vis = self.imgridfull(imgs, 2, 2)
            cv2.imshow("TUT Demo Visualizer", self.vis)
        elif self.vis1.mode == 3:
            if self.last_mode != self.vis1.mode:
                self.vis = frame.copy()
            else:
                self.vis = self.last_detected_image.copy()
            self.put_corner(frame, self.vis)
            cv2.imshow("TUT Demo Visualizer", self.vis)
        elif self.vis1.mode == 4:
            if self.last_mode != self.vis1.mode:
                self.vis = frame.copy()
            else:
                self.vis = self.max_detected_image.copy()
            self.put_corner(frame, self.vis)
            cv2.imshow("TUT Demo Visualizer", self.vis)

        self.vis1.process_keyboard(status, frame)
        self.vis1.previous_detections = self.vis1.number_detections
        self.last_mode = self.vis1.mode

    def imgridfull(self, imgs, rows, cols):
        h, w = imgs[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            r = i // cols
            c = i % cols
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        return grid

    def put_corner(self, src, dst):
        h, w = src.shape[:2]
        dst[:h, :w] = src

class UTIADemoVisualizer(Visualizer):
    def __init__(self, status):
        self.vis1 = DemoVisualizer(status)
        self.vis1.mode = 2
        self.vis = None
        self.temp_debug = None
        self.temp2_debug = None
        self.offset_from_max = 0
        self.last_detected_image = None
        self.max_detected_image = None
        self.last_mode = -1

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.vis1.process(status, frame, algorithm)
        if self.last_detected_image is None:
            self.last_detected_image = frame.copy()
        if self.max_detected_image is None:
            self.max_detected_image = frame.copy()
        if self.vis1.previous_detections * self.vis1.number_detections > 0:
            self.last_detected_image = self.vis1.vis.copy()
        if self.vis1.offset_from_max_detection == 0:
            self.max_detected_image = self.vis1.vis.copy()
            self.offset_from_max = 0
        else:
            self.offset_from_max += 1

        if self.vis1.mode == 2:
            status.window.vis_table = True
        else:
            status.window.vis_table = False

        player_name = status.input_string
        string_size = 10
        if len(player_name) < string_size:
            player_name = player_name + ' ' * (string_size - len(player_name))
        if len(player_name) > string_size:
            player_name = player_name[:string_size]

        sp_now = round(self.vis1.max_speed * 100) / 100
        put_it_there = True
        for el in status.window.table:
            if abs(el[0] - sp_now) < 0.001 and el[1] == player_name:
                put_it_there = False
                break

        if put_it_there:
            if len(status.window.table) < 10:
                status.window.table.append((sp_now, player_name))
            else:
                last_el = status.window.table[-1]
                if sp_now > last_el[0]:
                    status.window.table[-1] = (sp_now, player_name)
            status.window.table.sort(reverse=True)

        if status.window.vis_table:
            for i, el in enumerate(status.window.table):
                pref = " " if i < 9 else ""
                status.window.print(pref + str(i + 1) + ". " + el[1] + ": " + str(el[0])[:4] + " km/h")

        cv2.imshow("UTIA Demo Visualizer", self.vis1.vis)
        self.vis1.process_keyboard(status, frame)
        self.vis1.previous_detections = self.vis1.number_detections
        self.last_mode = self.vis1.mode

class RemovalVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.frames = [None] * 5
        self.current = None
        self.output_cache = None
        self.object_points = []
        self.points_cache = None
        self.events_detected = 0
        self.manual = None

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.events_detected += 1
        self.stats.append(time.time())
        fps_estimate = len(self.stats) / (self.stats[-1] - self.stats[0])

        self.current = frame.copy()
        self.frames = [self.current] + self.frames[:-1]

        vis = self.frames[min(0 - algorithm.get_output_offset(), self.events_detected - 1)]
        bg = self.frames[min(self.events_detected - 1, 4)]

        self.output_cache = algorithm.get_output(False)
        self.object_points = [detection.get_points() for detection in self.output_cache.detections]
        self.points_cache = self.merge_point_sets(self.object_points)

        self.remove_points(self.points_cache, vis, bg)

        if self.manual:
            self.manual.frame(vis)

        cv2.imshow("Removal Visualizer", vis)
        self.process_keyboard(status, frame)

    def process_keyboard(self, status, frame):
        step = False
        while status.paused and not step and not status.quit:
            command = status.window.get_command(status.paused)
            if command == "PAUSE":
                status.paused = not status.paused
            if command == "STEP":
                step = True
            if command == "QUIT":
                status.quit = True
            if command == "SCREENSHOT":
                cv2.imwrite("screenshot.png", self.current)
            if command == "RECORD":
                if self.manual:
                    self.manual = None
                else:
                    self.manual = ManualRecorder(status.args.record_dir, frame.format, frame.shape[:2], fps_estimate)

            if not status.have_camera():
                if command == "JUMP_BACKWARD":
                    status.paused = False
                    status.args.frame = max(1, status.in_frame_num - 10)
                    status.reload = True
                if command == "JUMP_FORWARD":
                    status.paused = False
                    status.args.frame = status.in_frame_num + 10

    def merge_point_sets(self, point_sets):
        merged = []
        for point_set in point_sets:
            merged.extend(point_set)
        return merged

    def remove_points(self, points, vis, bg):
        for point in points:
            vis[point[1], point[0]] = bg[point[1], point[0]]

def process_video(status, input_num):
    input_video = VideoInput.make_from_file(status.args.inputs[input_num]) if not status.have_camera() else VideoInput.make_from_camera(status.args.camera)
    if status.args.exposure != 100:
        input_video.set_exposure(status.args.exposure)
    if status.args.fps != -1:
        input_video.set_fps(status.args.fps)
    dims = input_video.dims()
    fps = input_video.fps()
    status.input_name = os.path.basename(status.args.inputs[input_num]) if not status.have_camera() else "camera " + str(status.args.camera)

    evaluator = None
    if status.args.gts:
        evaluator = Evaluator(status.args.gts[input_num], dims, status.results, status.baseline)
    elif status.args.gt_dir:
        gt_path = os.path.join(status.args.gt_dir, status.args.names[input_num] + ".txt")
        evaluator = Evaluator(gt_path, dims, status.results, status.baseline)

    sequence_report = None
    if status.rpt:
        sequence_report = status.rpt.make_sequence(status.input_name)

    if not status.have_camera():
        wait_sec = status.args.wait / 1000.0 if status.have_wait() else 1.0 / fps
        status.window.set_frame_time(wait_sec)

    format = status.args.yuv and "YUV" or "BGR"
    object_vec = [None]
    algorithm = Algorithm.make(status.args.params, format, dims)
    frame = None
    frame_copy = None
    output_cache = None
    eval_result = None
    stat = Statistics()

    while not status.quit and not status.reload:
        allow_new_frames = True
        if evaluator:
            num_gt_frames = evaluator.gt().num_frames()
            if status.out_frame_num > num_gt_frames:
                break
            elif status.in_frame_num > num_gt_frames:
                allow_new_frames = False

        if allow_new_frames:
            frame = input_video.receive_frame()
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
                if status.args.pause_fn and eval_result.eval[3] > 0:
                    status.paused = True
                if status.args.pause_fp and eval_result.eval[2] > 0:
                    status.paused = True
                if status.args.pause_rg and eval_result.comp == "REGRESSION":
                    status.paused = True
                if status.args.pause_im and eval_result.comp == "IMPROVEMENT":
                    status.paused = True
            else:
                eval_result = EvalResult()
                eval_result.comp = "BUFFERING"

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
    input_video.default_camera()
    return stat

def main():
    parser = argparse.ArgumentParser(description="Fast Moving Objects Detection")
    parser.add_argument("--input", type=str, nargs="+", help="Input video files")
    parser.add_argument("--camera", type=int, default=-1, help="Camera ID")
    parser.add_argument("--exposure", type=float, default=100, help="Camera exposure")
    parser.add_argument("--fps", type=float, default=-1, help="Camera FPS")
    parser.add_argument("--wait", type=int, default=-1, help="Wait time between frames in ms")
    parser.add_argument("--frame", type=int, default=-1, help="Frame number to pause at")
    parser.add_argument("--yuv", action="store_true", help="Use YUV color space")
    parser.add_argument("--record_dir", type=str, default=".", help="Directory to save recordings")
    parser.add_argument("--gt_dir", type=str, help="Directory with ground truth files")
    parser.add_argument("--gts", type=str, nargs="+", help="Ground truth files")
    parser.add_argument("--names", type=str, nargs="+", help="Names of input files")
    parser.add_argument("--params", type=str, help="Algorithm parameters")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for evaluation")
    parser.add_argument("--pause_fn", action="store_true", help="Pause on false negatives")
    parser.add_argument("--pause_fp", action="store_true", help="Pause on false positives")
    parser.add_argument("--pause_rg", action="store_true", help="Pause on regressions")
    parser.add_argument("--pause_im", action="store_true", help="Pause on improvements")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--no_record", action="store_true", help="Disable recording")
    args = parser.parse_args()

    status = Status(args)
    if args.input:
        for input_num in range(len(args.input)):
            process_video(status, input_num)
    elif args.camera != -1:
        process_video(status, 0)

if __name__ == "__main__":
    main()
