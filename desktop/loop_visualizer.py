import cv2
import numpy as np
from collections import deque
from datetime import datetime

class Visualizer:
    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        raise NotImplementedError

class DebugVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.level = 1
        self.show_lm = False
        self.show_im = True
        self.add = 0
        self.vis = None
        self.detections = 0
        self.pause_first = False
        self.previous_det = 0

    def process(self, status, frame, evaluator, eval_result, algorithm):
        self.stats.append(datetime.now())
        fps_last = len(self.stats) / (self.stats[-1] - self.stats[0]).total_seconds()

        self.vis = algorithm.get_debug_image(self.level, self.show_im, self.show_lm, self.add)
        status.window.print(status.input_name)
        status.window.print(f"frame: {status.in_frame_num}")
        status.window.print(f"fps: {fps_last:.2f}")

        algorithm.get_output(self.output_cache, False)
        self.object_points = []
        for detection in self.output_cache.detections:
            points = detection.get_points()
            self.object_points.append(points)
        self.detections = len(self.output_cache.detections)

        if evaluator:
            status.window.print(eval_result.str())
            gt = evaluator.gt().get(status.out_frame_num)
            self.points_cache = self.merge_points(self.object_points)
            self.gt_points_cache = self.merge_points(gt)
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
            if command == "PAUSE_FIRST":
                self.pause_first = not self.pause_first
            if self.pause_first and self.detections > 0 and self.previous_det == self.detections:
                status.paused = not status.paused
                self.previous_det = 0
                self.detections = 0
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
            self.previous_det = self.detections

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.process(status, frame, evaluator, eval_result, algorithm)
        status.window.display(self.vis)
        self.process_keyboard(status, frame)

    def merge_points(self, points_list):
        merged_points = []
        for points in points_list:
            merged_points.extend(points)
        return merged_points

    def draw_points_gt(self, points, gt_points, image):
        for point in points:
            cv2.circle(image, point, 2, (255, 0, 255), -1)
        for point in gt_points:
            cv2.circle(image, point, 2, (0, 255, 0), -1)

    def draw_points(self, points, image, color):
        for point in points:
            cv2.circle(image, point, 2, color, -1)

    def good(self, eval):
        return eval[0] == 0 and eval[1] == 0

class UTIADemoVisualizer(Visualizer):
    def __init__(self, status):
        self.vis1 = DemoVisualizer(status)
        status.window.set_top_line("Fast Moving Objects Detection")
        self.vis1.mode = 2
        self.last_detected_image = None
        self.max_detected_image = None
        self.previous_detections = 0
        self.offset_from_max = 0
        self.last_mode = -1

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.vis1.process(status, frame, algorithm)
        if self.last_detected_image is None:
            self.last_detected_image = frame.copy()
        if self.max_detected_image is None:
            self.max_detected_image = frame.copy()
        if self.previous_detections * self.vis1.m_number_detections > 0:
            self.last_detected_image = self.vis1.vis.copy()
        if self.vis1.m_offset_from_max_detection == 0:
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
            player_name = player_name + " " * (string_size - len(player_name))
        if len(player_name) > string_size:
            player_name = player_name[:string_size]

        status.window.display(self.vis1.vis)
        self.vis1.process_keyboard(status, frame)
        self.previous_detections = self.vis1.m_number_detections
        self.last_mode = self.vis1.mode

class TUTDemoVisualizer(Visualizer):
    def __init__(self, status):
        self.vis1 = DemoVisualizer(status)
        self.input = []
        try:
            self.input.append(VideoInput.make_from_file("../data/webcam/circle_back_res.avi"))
            self.input.append(VideoInput.make_from_file("../data/webcam/counting_all_res.avi"))
            self.input.append(VideoInput.make_from_file("../data/webcam/floorball_res.avi"))
        except Exception:
            self.input.clear()
            try:
                self.input.append(VideoInput.make_from_file("data/webcam/circle_back_res.avi"))
                self.input.append(VideoInput.make_from_file("data/webcam/counting_all_res.avi"))
                self.input.append(VideoInput.make_from_file("data/webcam/floorball_res.avi"))
            except Exception:
                pass
        status.window.set_top_line("Fast Moving Objects Detection")
        self.record = not status.args.no_record
        if self.record:
            self.vis1.m_record_annotations = False
        self.vis = None
        self.temp_debug = None
        self.temp2_debug = None
        self.offset_from_max = 0
        self.last_detected_image = None
        self.max_detected_image = None
        self.last_mode = -1

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        if not self.vis1.m_manual and self.record:
            self.vis1.m_manual = ManualRecorder(status.args.record_dir, frame.format(), frame.dims(), 30)
        self.vis1.process(status, frame, algorithm)
        if self.last_detected_image is None:
            self.last_detected_image = frame.copy()
        if self.max_detected_image is None:
            self.max_detected_image = frame.copy()
        if self.previous_detections * self.vis1.m_number_detections > 0:
            self.last_detected_image = self.vis1.vis.copy()
        if self.vis1.m_offset_from_max_detection == 0:
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
            for _ in range(len(imgs), 4):
                imgs.append(self.vis1.vis)
            self.vis = self.imgridfull(imgs, 2, 2)
            status.window.display(self.vis)
        elif self.vis1.mode == 1:
            status.window.display(self.vis1.vis)
        elif self.vis1.mode == 2:
            self.vis = np.zeros((2 * frame.shape[0], 2 * frame.shape[1], 3), dtype=np.uint8)
            imgs = [frame]
            temp = algorithm.get_debug_image(1, True, False, 1)
            temp = cv2.resize(temp, (frame.shape[1], frame.shape[0]))
            imgs.append(temp)
            temp2 = algorithm.get_debug_image(1, True, True, 3)
            temp2 = cv2.resize(temp2, (frame.shape[1], frame.shape[0]))
            imgs.append(temp2)
            imgs.append(self.vis1.vis)
            self.vis = self.imgridfull(imgs, 2, 2)
            status.window.display(self.vis)
        elif self.vis1.mode == 3:
            if self.last_mode != self.vis1.mode:
                self.vis = frame.copy()
            else:
                self.vis = self.last_detected_image.copy()
            self.putcorner(frame, self.vis)
            status.window.display(self.vis)
        elif self.vis1.mode == 4:
            if self.last_mode != self.vis1.mode:
                self.vis = frame.copy()
            else:
                self.vis = self.max_detected_image.copy()
            self.putcorner(frame, self.vis)
            status.window.display(self.vis)
        self.vis1.process_keyboard(status, frame)
        self.previous_detections = self.vis1.m_number_detections
        self.last_mode = self.vis1.mode

    def imgridfull(self, imgs, rows, cols):
        h, w = imgs[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for i, img in enumerate(imgs):
            r = i // cols
            c = i % cols
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        return grid

    def putcorner(self, src, dst):
        h, w = src.shape[:2]
        dst[:h, :w] = src

class DemoVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.show_help = False
        self.automatic = None
        self.manual = None
        self.record_annotations = True
        self.forced_event = False
        self.output = None
        self.segments = []
        self.curves = []
        self.events_detected = 0
        self.max_detections = 0
        self.last_detect_frame = -12
        self.vis = None
        self.mode = 0
        self.number_detections = 0
        self.offset_from_max_detection = 0
        self.speeds = []
        self.last_speeds = []
        self.max_speed = 0
        self.update_help(status)

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

        status.window.print(f"Detections: {self.max_detections}")
        for i, speed in enumerate(self.speeds):
            status.window.print(f"Speed {i + 1} : {round(speed * fctr * 100) / 100:.2f}{meas}")
        status.window.print(f"Max speed: {round(self.max_speed * fctr * 100) / 100:.2f}{meas}", (255, 0, 0))

        status.window.set_text_color((255, 0, 0) if recording else (192, 192, 192))

    def on_detection(self, status, detection):
        if status.out_frame_num - self.last_detect_frame > 12:
            if status.sound:
                print('\a', end='')
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
        for i, (time, speed) in enumerate(self.last_speeds):
            if time > 0:
                self.last_speeds[i] = (time - 1, speed)
                if speed > speed_now:
                    speed_now = speed

        if detection.object.curve:
            self.max_speed = speed_now

        if len(self.segments) > 200:
            self.segments = self.segments[len(self.segments) // 2:]

        while len(self.curves) > 20:
            self.curves.pop(0)

    def draw_segments(self, image):
        color = (255, 0, 255)
        thickness = 8
        for segment in self.segments:
            color = (min(color[0] + 2, 255), min(color[1] + 1, 255), min(color[2] + 4, 255))
            cv2.line(image, segment[0], segment[1], color, thickness)
        for curve in self.curves:
            color = (min(color[0] + 2, 255), min(color[1] + 1, 255), min(color[2] + 4, 255))
            curve.draw_smooth(image, color, thickness)

    def process(self, status, frame, algorithm):
        self.stats.append(datetime.now())
        fps_estimate = len(self.stats) / (self.stats[-1] - self.stats[0]).total_seconds()

        algorithm.get_output(self.output, True)
        self.number_detections = len(self.output.detections)
        if self.number_detections > 0:
            self.speeds.clear()

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
            event = self.forced_event or bool(self.output.detections)
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
                n_players = 10
                string_size = 10
                vis_table = status.window.vis_table
                status.window.vis_table = False
                sp_now = round(self.max_speed * 100) / 100

                if len(status.window.m_table) >= n_players:
                    last_el = status.window.m_table[-1]
                    if sp_now <= last_el[0]:
                        status.window.set_center_line(f"Sorry, not in top {n_players}", "")
                        status.window.display(self.vis)
                        cv2.waitKey(500)
                        status.window.vis_table = vis_table
                        status.window.set_center_line("", "")
                        status.window.display(self.vis)
                        continue

                status.window.set_center_line("Input player's name", "")
                status.window.display(self.vis)

                key_code = ' '
                vec = []
                while key_code not in (13, '\n', 10):
                    if key_code != ' ':
                        if key_code in (127, 8):
                            if vec:
                                vec.pop()
                        else:
                            vec.append(key_code)
                        str_input = ''.join(vec)
                        status.window.set_center_line("Input player's name", str_input)
                        status.window.display(self.vis)
                    key_code = cv2.waitKey(0)
                status.window.set_center_line("", "")
                status.window.display(self.vis)
                str_input = ''.join(vec)
                status.input_string = str_input

                player_name = status.input_string
                if len(player_name) < string_size:
                    player_name = player_name + " " * (string_size - len(player_name))
                if len(player_name) > string_size:
                    player_name = player_name[:string_size]

                if len(status.window.m_table) < n_players:
                    status.window.m_table.append((sp_now, player_name))
                else:
                    last_el = status.window.m_table[-1]
                    if sp_now > last_el[0]:
                        status.window.m_table[-1] = (sp_now, player_name)
                status.window.m_table.sort(reverse=True)
                status.window.vis_table = vis_table
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
                    self.automatic = AutomaticRecorder(status.args.record_dir, frame.format(), frame.dims(), 30)
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
                    self.manual = ManualRecorder(status.args.record_dir, frame.format(), frame.dims(), 30)
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

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.process(status, frame, algorithm)
        status.window.display(self.vis)
        self.process_keyboard(status, frame)

class RemovalVisualizer(Visualizer):
    def __init__(self, status):
        self.stats = deque(maxlen=60)
        self.frames = [None] * 5
        self.current = None
        self.output_cache = None
        self.object_points = []
        self.points_cache = []
        self.events_detected = 0
        self.manual = None

    def visualize(self, status, frame, evaluator, eval_result, algorithm):
        self.events_detected += 1
        self.stats.append(datetime.now())
        fps_estimate = len(self.stats) / (self.stats[-1] - self.stats[0]).total_seconds()

        self.current = frame.copy()
        self.frames[4], self.frames[3], self.frames[2], self.frames[1], self.frames[0] = self.frames[3], self.frames[2], self.frames[1], self.frames[0], self.current

        vis = self.frames[min(0 - algorithm.get_output_offset(), self.events_detected - 1)]
        bg = self.frames[min(self.events_detected - 1, 4)]

        algorithm.get_output(self.output_cache, False)
        self.object_points = [detection.get_points() for detection in self.output_cache.detections]
        self.points_cache = self.merge_points(self.object_points)

        self.remove_points(self.points_cache, vis, bg)

        if self.manual:
            self.manual.frame(vis)

        status.window.display(vis)

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
                cv2.imwrite("screenshot.png", vis)

            if not status.have_camera():
                if command == "JUMP_BACKWARD":
                    status.paused = False
                    status.args.frame = max(1, status.in_frame_num - 10)
                    status.reload = True
                if command == "JUMP_FORWARD":
                    status.paused = False
                    status.args.frame = status.in_frame_num + 10
            if command == "RECORD":
                if self.manual:
                    self.manual = None
                else:
                    self.manual = ManualRecorder(status.args.record_dir, frame.format(), frame.dims(), fps_estimate)

    def merge_points(self, points_list):
        merged_points = []
        for points in points_list:
            merged_points.extend(points)
        return merged_points

    def remove_points(self, points, vis, bg):
        for point in points:
            vis[point[1], point[0]] = bg[point[1], point[0]]
