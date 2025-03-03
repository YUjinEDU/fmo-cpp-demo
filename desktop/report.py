import os
import fmo
from datetime import datetime

class EvaluationReport:
    def __init__(self, results, baseline, args, date, seconds):
        self.date = date
        self.results = results
        self.stats = self.Stats()
        self.info = self.generate_info(results, baseline, args, date, seconds)

    def write(self, out):
        out.write(self.info)

    def save(self, directory):
        if not self.results.list:
            return
        filename = os.path.join(directory, f"{self.date.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filename, 'w') as out:
            self.write(out)
            out.write('\n')
            self.results.save(out)

    def save_score(self, file):
        with open(file, 'w') as out:
            for i in range(self.Stats.NUM_STATS):
                out.write(f"{self.stats.avg[i]:.12f}\n")
            for i in range(self.Stats.NUM_STATS):
                out.write(f"{self.stats.total[i]:.12f}\n")
            out.write(f"{self.stats.iou:.12f}\n")

    def generate_info(self, results, baseline, args, date, seconds):
        fields = []
        have_base = False
        count = fmo.Evaluation()
        count_base = fmo.Evaluation()
        sum_stats = [0] * self.Stats.NUM_STATS
        sum_base_stats = [0] * self.Stats.NUM_STATS
        num_files = 0
        num_base_files = 0

        def precision(count):
            if count[fmo.Event.FP] == 0:
                return 1.0
            div = count[fmo.Event.TP] + count[fmo.Event.FP]
            return count[fmo.Event.TP] / div

        def recall(count):
            if count[fmo.Event.FN] == 0:
                return 1.0
            div = count[fmo.Event.TP] + count[fmo.Event.FN]
            return count[fmo.Event.TP] / div

        def fscore(count, beta):
            p = precision(count)
            r = recall(count)
            if p <= 0 or r <= 0:
                return 0.0
            beta_sqr = beta * beta
            return ((beta_sqr + 1) * p * r) / ((beta_sqr * p) + r)

        def fscore_05(count):
            return fscore(count, 0.5)

        def fscore_10(count):
            return fscore(count, 1.0)

        def fscore_20(count):
            return fscore(count, 2.0)

        def percent(out, val):
            out.write(f"{val * 100:.2f}%")

        stat_funcs = [precision, recall, fscore_05, fscore_10, fscore_20]
        func_names = ["precision", "recall", "F_0.5", "F_1.0", "F_2.0"]
        func_displayed = [True, True, False, True, False]

        def count_str_impl(val, val_base):
            delta = val - val_base
            return f"{val} ({delta:+d})" if delta != 0 else f"{val}"

        def percent_str_impl(val, val_base):
            delta = val - val_base
            return f"{val * 100:.2f}% ({delta * 100:+.2f}%)" if abs(delta) > 5e-5 else f"{val * 100:.2f}%"

        def count_str(event):
            val = count[event]
            val_base = count_base[event] if have_base else val
            return count_str_impl(val, val_base)

        def percent_str(i):
            val = stat_funcs[i](count)
            val_base = stat_funcs[i](count_base) if have_base else val
            return percent_str_impl(val, val_base)

        def add_to_average(i):
            sum_stats[i] += stat_funcs[i](count)
            if have_base:
                sum_base_stats[i] += stat_funcs[i](count_base)

        fields.extend(["sequence", "tp", "tn", "fp", "fn"])
        for i in range(self.Stats.NUM_STATS):
            if func_displayed[i]:
                fields.append(func_names[i])

        for file in results.list:
            if not file.frames:
                continue
            base_file = baseline.get_file(file.name)
            have_base = len(base_file.frames) == len(file.frames)

            count.clear()
            for eval in file.frames:
                count += eval
            if have_base:
                count_base.clear()
                for eval in base_file.frames:
                    count_base += eval

            name = args.names[num_files] if args.names else file.name
            fields.extend([name, count_str(fmo.Event.TP), count_str(fmo.Event.TN), count_str(fmo.Event.FP), count_str(fmo.Event.FN)])
            for i in range(self.Stats.NUM_STATS):
                if func_displayed[i]:
                    fields.append(percent_str(i))
                add_to_average(i)

            num_files += 1
            if have_base:
                num_base_files += 1

        if num_files == 0:
            return ""

        count.clear()
        count_base.clear()
        for file in results.list:
            if not file.frames:
                continue
            base_file = baseline.get_file(file.name)
            have_base = len(base_file.frames) == len(file.frames)

            for eval in file.frames:
                count += eval
            if have_base:
                for eval in base_file.frames:
                    count_base += eval

        have_base = num_base_files > 0

        fields.extend(["total", count_str(fmo.Event.TP), count_str(fmo.Event.TN), count_str(fmo.Event.FP), count_str(fmo.Event.FN)])
        for i in range(self.Stats.NUM_STATS):
            self.stats.total[i] = stat_funcs[i](count)
            self.stats.total_base[i] = stat_funcs[i](count_base) if have_base else self.stats.total[i]
            if func_displayed[i]:
                fields.append(percent_str_impl(self.stats.total[i], self.stats.total_base[i]))

        fields.extend(["average", "", "", "", ""])
        for i in range(self.Stats.NUM_STATS):
            self.stats.avg[i] = sum_stats[i] / num_files
            self.stats.avg_base[i] = sum_base_stats[i] / num_files if have_base else self.stats.avg[i]
            if func_displayed[i]:
                fields.append(percent_str_impl(self.stats.avg[i], self.stats.avg_base[i]))

        cols = 5 + sum(func_displayed)
        col_size = [0] * cols

        def hline(out):
            for i in range(col_size[0]):
                out.write('-')
            for col in range(1, cols):
                out.write('|')
                for i in range(col_size[col]):
                    out.write('-')
            out.write('\n')

        for i in range(0, len(fields), cols):
            for col in range(cols):
                col_size[col] = max(col_size[col], len(fields[i + col]) + 1)

        num_bins = 10
        hist = results.make_iou_histogram(num_bins)
        hist_base = baseline.make_iou_histogram(num_bins)
        self.stats.iou = results.get_average_iou()
        self.stats.iou_base = baseline.get_average_iou() if have_base else self.stats.iou

        out = []
        out.append(f"parameters: {args.parameters()}\n")
        out.append(f"generated on: {date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.append(f"evaluation time: {seconds:.1f} s\n")
        out.append("iou: ")
        for i in range(num_bins):
            bin_val = hist[i]
            bin_base = hist_base[i] if have_base else bin_val
            out.append(count_str_impl(bin_val, bin_base) + " ")
        out.append('\n')
        out.append(f"iou avg: {percent_str_impl(self.stats.iou, self.stats.iou_base)}\n")
        for i in range(self.Stats.NUM_STATS):
            if not func_displayed[i]:
                out.append(f"{func_names[i]} total: {percent_str_impl(self.stats.total[i], self.stats.total_base[i])}, avg: {percent_str_impl(self.stats.avg[i], self.stats.avg_base[i])}\n")
        out.append('\n')

        for i in range(0, len(fields), cols):
            out.append(f"{fields[i]:<{col_size[0]}}")
            for col in range(1, cols):
                out.append(f"|{fields[i + col]:<{col_size[col]}}")
            out.append('\n')
            if i == 0 or i == num_files * cols:
                hline(out)

        return ''.join(out)

    class Stats:
        NUM_STATS = 5

        def __init__(self):
            self.avg = [0] * self.NUM_STATS
            self.total = [0] * self.NUM_STATS
            self.iou = 0
            self.avg_base = [0] * self.NUM_STATS
            self.total_base = [0] * self.NUM_STATS
            self.iou_base = 0

class DetectionReport:
    def __init__(self, directory, date):
        self.out = open(self.file_name(directory, date), 'w')
        self.out.write('<?xml version="1.0" ?>\n')
        self.out.write('<run>\n')
        self.out.write(f"  <date>{date.strftime('%Y-%m-%d %H:%M:%S')}</date>\n")

    def __del__(self):
        self.out.write('</run>\n')
        self.out.close()

    def make_sequence(self, input):
        return self.Sequence(self, input)

    def file_name(self, directory, date):
        return os.path.join(directory, f"{date.strftime('%Y%m%d_%H%M%S')}.xml")

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
                    self.report.out.write('      <detection>\n')

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

                self.report.out.write('        <points>')
                points = detection.get_points()
                for p in points:
                    self.report.out.write(f"{p.x} {p.y} ")
                self.report.out.write('</points>\n')

                self.report.out.write('      </detection>\n')
            self.report.out.write('    </frame>\n')
