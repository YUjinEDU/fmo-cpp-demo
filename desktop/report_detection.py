import os
import fmo
from datetime import datetime

class DetectionReport:
    def __init__(self, directory, date):
        self.file_path = os.path.join(directory, f"{date.strftime('%Y%m%d_%H%M%S')}.xml")
        self.out = open(self.file_path, 'w')
        self.out.write('<?xml version="1.0" ?>\n')
        self.out.write('<run>\n')
        self.out.write(f"  <date>{date.strftime('%Y-%m-%d %H:%M:%S')}</date>\n")

    def __del__(self):
        self.out.write('</run>\n')
        self.out.close()

    def make_sequence(self, input):
        return self.Sequence(self, input)

    class Sequence:
        def __init__(self, report, input):
            self.report = report
            self.report.out.write(f"  <sequence input=\"{input}\">\n")

        def __del__(self):
            self.report.out.write('  </sequence>\n')

        def write_frame(self, frame_num, alg_out, eval_res):
            if not alg_out.detections:
                return
            self.report.out.write(f"    <frame num=\"{frame_num}\">\n")
            for i, detection in enumerate(alg_out.detections):
                if detection.object.have_id():
                    self.report.out.write(f"      <detection id=\"{detection.object.id}\">\n")
                else:
                    self.report.out.write("      <detection>\n")
                if detection.predecessor.have_id():
                    self.report.out.write(f"        <predecessor>{detection.predecessor.id}</predecessor>\n")
                if detection.object.have_center():
                    self.report.out.write(f"        <center x=\"{detection.object.center.x}\" y=\"{detection.object.center.y}\"/>\n")
                if detection.object.have_direction():
                    self.report.out.write(f"        <direction x=\"{detection.object.direction[0]}\" y=\"{detection.object.direction[1]}\"/>\n")
                if detection.object.have_length():
                    self.report.out.write(f"        <length unit=\"px\">{detection.object.length}</length>\n")
                if detection.object.have_radius():
                    self.report.out.write(f"        <radius unit=\"px\">{detection.object.radius}</radius>\n")
                if detection.object.have_velocity():
                    self.report.out.write(f"        <velocity unit=\"px/frame\">{detection.object.velocity}</velocity>\n")
                if len(eval_res.iou_dt) > i:
                    self.report.out.write(f"        <iou>{eval_res.iou_dt[i]}</iou>\n")
                self.report.out.write("        <points>")
                points = detection.get_points()
                for p in points:
                    self.report.out.write(f"{p.x} {p.y} ")
                self.report.out.write("</points>\n")
                self.report.out.write("      </detection>\n")
            self.report.out.write("    </frame>\n")
