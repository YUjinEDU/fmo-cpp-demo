import argparse
import os
import sys
import json

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Fast Moving Objects Detection")
        self.add_arguments()
        self.args = self.parser.parse_args()

    def add_arguments(self):
        self.parser.add_argument("--help", action="help", help="Display help.")
        self.parser.add_argument("--defaults", action="store_true", help="Display default values for all parameters.")
        self.parser.add_argument("--algorithm", type=str, help="<name> Specifies the name of the algorithm variant. Use --list to list available algorithm names.")
        self.parser.add_argument("--list", action="store_true", help="Display available algorithm names. Use --algorithm to select an algorithm.")
        self.parser.add_argument("--headless", action="store_true", help="Don't draw any GUI unless the playback is paused. Must not be used with --wait, --fast.")
        self.parser.add_argument("--demo", action="store_true", help="Force demo visualization method. This visualization method is preferred when --camera is used.")
        self.parser.add_argument("--debug", action="store_true", help="Force debug visualization method. This visualization method is preferred when --input is used.")
        self.parser.add_argument("--removal", action="store_true", help="Force removal visualization method. This visualization method has the highest priority.")
        self.parser.add_argument("--no-record", action="store_true", help="Switch off record all frames.")
        self.parser.add_argument("--include", type=str, help="<path> File with additional command-line arguments. The format is the same as when specifying parameters on the command line. Whitespace such as tabs and endlines is allowed.")
        self.parser.add_argument("--input", type=str, nargs='+', help="<path> Path to an input video file. Can be used multiple times. Must not be used with --camera.")
        self.parser.add_argument("--input-dir", type=str, help="<path> Path to an input videos directory. Must not be used with --camera. Provides a template for input videos. Asterisk (*) will be replaced with input. In case of no input all sequences in list.txt (in the directory) will be used")
        self.parser.add_argument("--gt", type=str, nargs='+', help="<path> Text file containing ground truth data. Using this option enables quality evaluation. If used at all, this option must be used as many times as --input. Use --eval-dir to specify the directory for evaluation results.")
        self.parser.add_argument("--gt-dir", type=str, help="<path> Directory with text files containing ground truth data. Using this option enables quality evaluation. Use --eval-dir to specify the directory for evaluation results.")
        self.parser.add_argument("--name", type=str, nargs='+', help="<string> Name of the input file to be displayed in the evaluation report. If used at all, this option must be used as many times as --input.")
        self.parser.add_argument("--baseline", type=str, help="<path> File with previously saved results (via --eval-dir) for comparison. When used, the playback will pause to demonstrate where the results differ. Must be used with --gt.")
        self.parser.add_argument("--camera", type=int, help="<int> Input camera device ID. When this option is used, stream from the specified camera will be used as input. Using ID 0 selects the default camera, if available. Must not be used with --input, --wait, --fast, --frame, --pause.")
        self.parser.add_argument("--yuv", action="store_true", help="Feed image data into the algorithm in YCbCr color space.")
        self.parser.add_argument("--record-dir", type=str, help="<dir> Output directory to save video to. A new video file will be created, storing the input video with optionally overlaid detections. The name of the video file will be determined by system time. The directory must exist.")
        self.parser.add_argument("--eval-dir", type=str, help="<dir> Directory to save evaluation report to. A single file text file will be created there with a unique name based on timestamp. Must be used with --gt.")
        self.parser.add_argument("--tex", action="store_true", help="Format tables in the evaluation report so that they can be used in the TeX typesetting system. Must be used with --eval-dir.")
        self.parser.add_argument("--detect-dir", type=str, help="<dir> Directory to save detection output to. A single XML file will be created there with a unique name based on timestamp.")
        self.parser.add_argument("--score-file", type=str, help="<file> File to write a numeric evaluation score to.")
        self.parser.add_argument("--pause-fp", action="store_true", help="Playback will pause whenever a detection is deemed a false positive. Must be used with --gt.")
        self.parser.add_argument("--pause-fn", action="store_true", help="Playback will pause whenever a detection is deemed a false negative. Must be used with --gt.")
        self.parser.add_argument("--pause-rg", action="store_true", help="Playback will pause whenever a regression is detected, i.e. whenever a frame is evaluated as false and baseline is true. Must be used with --baseline.")
        self.parser.add_argument("--pause-im", action="store_true", help="Playback will pause whenever an improvement is detected, i.e. whenever a frame is evaluated as true and baseline is false. Must be used with --baseline.")
        self.parser.add_argument("--paused", action="store_true", help="Playback will be paused on the first frame. Shorthand for --frame 1. Must not be used with --camera.")
        self.parser.add_argument("--frame", type=int, help="<frame> Playback will be paused on the specified frame number. If there are multiple input files, playback will be paused in each file that contains a frame with the specified frame number. Must not be used with --camera.")
        self.parser.add_argument("--fast", action="store_true", help="Sets the maximum playback speed. Shorthand for --wait 0. Must not be used with --camera, --headless.")
        self.parser.add_argument("--wait", type=int, help="<ms> Specifies the frame time in milliseconds, allowing for slow playback. Must not be used with --camera, --headless.")
        self.parser.add_argument("--exposure", type=float, help="Set exposure value. Should be between 0 and 1. Usually between 0.03 and 0.1.")
        self.parser.add_argument("--fps", type=float, help="Set number of frames per second.")
        self.parser.add_argument("--radius", type=float, help="Set object radius in cm. Used for speed estimation. Used if --p2cm is not specified. By default used for tennis/floorball: 3.6 cm.")
        self.parser.add_argument("--p2cm", type=float, help="Set how many cm are in one pixel on object. Used for speed estimation. More dominant than --radius.")
        self.parser.add_argument("--dfactor", type=float, help="Differential image threshold factor. Default 1.0.")
        self.parser.add_argument("--p-iou-thresh", type=float, help="<float>")
        self.parser.add_argument("--p-diff-thresh", type=int, help="<uint8>")
        self.parser.add_argument("--p-diff-adjust-period", type=int, help="<int>")
        self.parser.add_argument("--p-diff-min-noise", type=float, help="<float>")
        self.parser.add_argument("--p-diff-max-noise", type=float, help="<float>")
        self.parser.add_argument("--p-max-gap-x", type=float, help="<float>")
        self.parser.add_argument("--p-min-gap-y", type=float, help="<float>")
        self.parser.add_argument("--p-max-image-height", type=int, help="<int>")
        self.parser.add_argument("--p-image-height", type=int, help="<int>")
        self.parser.add_argument("--p-min-strip-height", type=int, help="<int>")
        self.parser.add_argument("--p-min-strips-in-object", type=int, help="<int>")
        self.parser.add_argument("--p-min-strip-area", type=float, help="<float>")
        self.parser.add_argument("--p-min-aspect", type=float, help="<float>")
        self.parser.add_argument("--p-min-aspect-for-relevant-angle", type=float, help="<float>")
        self.parser.add_argument("--p-min-dist-to-t-minus-2", type=float, help="<float>")
        self.parser.add_argument("--p-match-aspect-max", type=float, help="<float>")
        self.parser.add_argument("--p-match-area-max", type=float, help="<float>")
        self.parser.add_argument("--p-match-distance-min", type=float, help="<float>")
        self.parser.add_argument("--p-match-distance-max", type=float, help="<float>")
        self.parser.add_argument("--p-match-angle-max", type=float, help="<float>")
        self.parser.add_argument("--p-match-aspect-weight", type=float, help="<float>")
        self.parser.add_argument("--p-match-area-weight", type=float, help="<float>")
        self.parser.add_argument("--p-match-distance-weight", type=float, help="<float>")
        self.parser.add_argument("--p-match-angle-weight", type=float, help="<float>")
        self.parser.add_argument("--p-select-max-distance", type=float, help="<float>")
        self.parser.add_argument("--p-output-radius-corr", type=float, help="<float>")
        self.parser.add_argument("--p-output-radius-min", type=float, help="<float>")
        self.parser.add_argument("--p-output-raster-corr", type=float, help="<float>")
        self.parser.add_argument("--p-output-no-robust-radius", action="store_true", help="<flag>")
        self.parser.add_argument("--p-min-strips-in-component", type=int, help="<int>")
        self.parser.add_argument("--p-min-strips-in-cluster", type=int, help="<int>")
        self.parser.add_argument("--p-min-cluster-length", type=float, help="<float>")
        self.parser.add_argument("--p-weight-height-ratio", type=float, help="<float>")
        self.parser.add_argument("--p-weight-distance", type=float, help="<float>")
        self.parser.add_argument("--p-weight-gaps", type=float, help="<float>")
        self.parser.add_argument("--p-max-height-ratio-internal", type=float, help="<float>")
        self.parser.add_argument("--p-max-height-ratio-external", type=float, help="<float>")
        self.parser.add_argument("--p-max-distance", type=float, help="<float>")
        self.parser.add_argument("--p-max-gaps-length", type=float, help="<float>")
        self.parser.add_argument("--p-min-motion", type=float, help="<float>")

    def validate(self):
        if not self.args.input and self.args.camera is None and not self.args.input_dir:
            raise ValueError("one of --input, --input-dir, --camera must be specified")
        if self.args.input_dir and self.args.input:
            raise ValueError("--input and --input-dir cannot be used together")
        if self.args.input and self.args.gt_dir:
            raise ValueError("--input and --gt-dir cannot be used together")
        if self.args.camera is not None:
            if self.args.input:
                raise ValueError("--camera cannot be used with --input")
            if self.args.wait is not None:
                raise ValueError("--camera cannot be used with --wait or --fast")
            if self.args.frame is not None:
                raise ValueError("--camera cannot be used with --frame or --pause")
        if self.args.gt:
            if len(self.args.gt) != len(self.args.input):
                raise ValueError("there must be one --gt for each --input")
        if not self.args.gt and not self.args.gt_dir:
            if self.args.pause_fn or self.args.pause_fp:
                raise ValueError("--pause-fn|fp must be used with --gt")
            if self.args.eval_dir:
                raise ValueError("--eval-dir must be used with --gt")
            if self.args.baseline:
                raise ValueError("--baseline must be used with --gt")
        if not self.args.baseline:
            if self.args.pause_rg or self.args.pause_im:
                raise ValueError("--pause-rg|im must be used with --baseline")
        if sum([self.args.removal, self.args.demo, self.args.debug, self.args.headless]) != 1:
            raise ValueError("One visualization method should be used.")
        if self.args.headless and self.args.wait is not None:
            raise ValueError("--headless cannot be used with --wait or --fast")
        if not self.args.eval_dir and self.args.tex:
            raise ValueError("--tex cannot be used without --eval-dir")

    def print_help(self):
        self.parser.print_help()

    def print_values(self):
        args_dict = vars(self.args)
        for key, value in args_dict.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    parser = Parser()
    parser.validate()
    parser.print_help()
    parser.print_values()
