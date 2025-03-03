import os
import sys
import time
import datetime
import cv2
import numpy as np
from args import Args
from evaluator import Evaluator, Results, extractFilename
from report import EvaluationReport, DetectionReport
from loop_visualizer import DebugVisualizer, DemoVisualizer, TUTDemoVisualizer, UTIADemoVisualizer, RemovalVisualizer
from video import VideoInput
from window import Window, Command
from fmo import Algorithm, TimeUnit, Timer

class Status:
    def __init__(self, argc, argv):
        self.args = Args(argc, argv)
        self.window = Window()
        self.results = Results()
        self.baseline = Results()
        self.date = datetime.datetime.now()
        self.timer = Timer()
        self.inputName = ""
        self.visualizer = None
        self.rpt = None
        self.inFrameNum = 0
        self.outFrameNum = 0
        self.paused = False
        self.quit = False
        self.reload = False
        self.sound = False
        self.inputString = "Baseline"

    def haveCamera(self):
        return self.args.camera != -1

    def haveWait(self):
        return self.args.wait != -1

    def haveFrame(self):
        return self.args.frame != -1

    def unsetFrame(self):
        self.args.frame = -1

def replace(string, old, new):
    return string.replace(old, new)

def printStatistics(stats):
    print("\nStatistics together:")
    totalSum = 0
    totalFrames = 0
    nDets = []
    for stat in stats:
        nDets.append(stat.totalDetections)
        totalSum += stat.totalDetections
        totalFrames += stat.nFrames
    meanFrames = totalSum / totalFrames
    meanSeq = totalSum / len(stats)
    meanNFrames = totalFrames / len(stats)
    median = sorted(nDets)[len(nDets) // 2]
    print(f"Number of sequences - {len(stats)}")
    print(f"Average number of frames - {meanNFrames}")
    print(f"Detections: total - {totalSum}, average per frame - {meanFrames}, average per sequence - {meanSeq}, median per sequence - {median}")

def processVideo(s, inputNum):
    if len(s.args.names) > inputNum:
        print(f"Processing {s.args.names[inputNum]}")
    input = VideoInput.makeFromFile(s.args.inputs[inputNum]) if not s.haveCamera() else VideoInput.makeFromCamera(s.args.camera)
    if s.args.exposure != 100:
        input.set_exposure(s.args.exposure)
    if s.args.fps != -1:
        input.set_fps(s.args.fps)
    print(f"Exposure value: {input.get_exposure()}")
    print(f"FPS: {input.fps()}")
    dims = input.dims()
    fps = input.fps()
    s.inputName = extractFilename(s.args.inputs[inputNum]) if not s.haveCamera() else f"camera {s.args.camera}"
    evaluator = None
    if s.args.gts:
        evaluator = Evaluator(s.args.gts[inputNum], dims, s.results, s.baseline)
    elif s.args.gtDir:
        evaluator = Evaluator(os.path.join(s.args.gtDir, f"{s.args.names[inputNum]}.txt"), dims, s.results, s.baseline)
    sequenceReport = s.rpt.makeSequence(s.inputName) if s.rpt else None
    if not s.haveCamera():
        waitSec = s.args.wait / 1e3 if s.haveWait() else 1 / fps
        s.window.setFrameTime(waitSec)
    format = Algorithm.Format.YUV if s.args.yuv else Algorithm.Format.BGR
    objectVec = [Algorithm.PointSet() for _ in range(1)]
    algorithm = Algorithm.make(s.args.params, format, dims)
    frame = None
    frameCopy = Algorithm.Image(format, dims)
    outputCache = Algorithm.Output()
    evalResult = Evaluator.EvalResult()
    s.inFrameNum = 1
    s.outFrameNum = 1 + algorithm.getOutputOffset()
    stat = Statistics()
    while not s.quit and not s.reload:
        allowNewFrames = True
        if evaluator:
            numGtFrames = evaluator.gt().numFrames()
            if s.outFrameNum > numGtFrames:
                break
            elif s.inFrameNum > numGtFrames:
                allowNewFrames = False
        if allowNewFrames:
            frame = input.receiveFrame()
            if frame.data() is None:
                break
            if s.haveCamera():
                Algorithm.flip(frame, frame)
        Algorithm.convert(frame, frameCopy, format)
        algorithm.setInputSwap(frameCopy)
        algorithm.getOutput(outputCache, False)
        stat.nextFrame(len(outputCache.detections))
        if evaluator:
            if s.outFrameNum >= 1:
                evaluator.evaluateFrame(outputCache, s.outFrameNum, evalResult, s.args.params.iouThreshold)
                if s.args.pauseFn and evalResult.eval[Evaluator.Event.FN] > 0:
                    s.paused = True
                if s.args.pauseFp and evalResult.eval[Evaluator.Event.FP] > 0:
                    s.paused = True
                if s.args.pauseRg and evalResult.comp == Evaluator.Comparison.REGRESSION:
                    s.paused = True
                if s.args.pauseIm and evalResult.comp == Evaluator.Comparison.IMPROVEMENT:
                    s.paused = True
            else:
                evalResult.clear()
                evalResult.comp = Evaluator.Comparison.BUFFERING
        if sequenceReport:
            sequenceReport.writeFrame(s.outFrameNum, outputCache, evalResult)
        if s.args.frame == s.inFrameNum:
            s.unsetFrame()
            s.paused = True
        if s.haveFrame():
            continue
        if s.args.headless and not s.paused:
            continue
        s.visualizer.visualize(s, frame, evaluator, evalResult, algorithm)
    stat.print()
    input.default_camera()
    return stat

def main(argc, argv):
    try:
        s = Status(argc, argv)
        if s.args.inputDir:
            if not s.args.names:
                path = os.path.join(s.args.inputDir.split("*")[0], "list.txt")
                with open(path, "r") as file:
                    s.args.names = [line.strip() for line in file]
            for name in s.args.names:
                s.args.inputs.append(s.args.inputDir.replace("*", name))
        if s.args.baseline:
            s.baseline.load(s.args.baseline)
        if s.haveCamera():
            s.args.inputs.append("")
        if s.args.detectDir:
            s.rpt = DetectionReport(s.args.detectDir, s.date)
        demo = s.haveCamera()
        if s.args.demo:
            demo = True
        if s.args.debug:
            demo = False
        if s.args.removal:
            s.visualizer = RemovalVisualizer(s)
        elif s.args.tutdemo:
            s.visualizer = TUTDemoVisualizer(s)
        elif s.args.utiademo:
            s.visualizer = UTIADemoVisualizer(s)
        else:
            s.visualizer = DemoVisualizer(s) if demo else DebugVisualizer(s)
        stats = [processVideo(s, i) for i in range(len(s.args.inputs)) if not s.quit]
        report = EvaluationReport(s.results, s.baseline, s.args, s.date, s.timer.toc(TimeUnit.SEC, float))
        report.write(sys.stdout)
        printStatistics(stats)
        if s.args.evalDir:
            report.save(s.args.evalDir)
        if s.args.scoreFile:
            report.saveScore(s.args.scoreFile)
    except Exception as e:
        print(f"error: {e}")
        print("tip: use --help to see a list of available commands")
        return -1

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
