import threading
import ctypes
from java_classes import Callback
from env import Env
import fmo

class Global:
    def __init__(self):
        self.mutex = threading.Lock()
        self.javaVM = None
        self.stop = False
        self.exchange = None
        self.callbackRef = None
        self.image = None
        self.dims = None
        self.format = None
        self.config = fmo.Algorithm.Config()

global_data = Global()

def statsString(stats):
    q = stats.quantilesMs()
    return f"{q.q50:.2f} / {q.q95:.1f} / {q.q99:.0f}"

def threadImpl():
    threadEnv = Env(global_data.javaVM, "Lib")
    env = threadEnv.get()
    frameStats = fmo.FrameStats()
    frameStats.reset(30)
    sectionStats = fmo.SectionStats()
    input_image = fmo.Image(global_data.format, global_data.dims)
    output = fmo.Algorithm.Output()
    explorer = fmo.Algorithm.make(global_data.config, global_data.format, global_data.dims)
    callback = global_data.callbackRef.get(env)
    callback.log("Detection started")

    while not global_data.stop:
        global_data.exchange.swapReceive(input_image)
        if global_data.stop:
            break

        frameStats.tick()
        sectionStats.start()
        explorer.setInputSwap(input_image)
        explorer.getOutput(output)
        statsUpdated = sectionStats.stop()

        if statsUpdated:
            stats = statsString(sectionStats)
            callback.log(stats)

        if output.detections:
            detections = DetectionArray(env, len(output.detections))
            for i, det in enumerate(output.detections):
                detection = Detection(env, det)
                detections.set(i, detection)
            callback.onObjectsDetected(detections)

    global_data.callbackRef.release(env)

def Java_cz_fmo_Lib_detectionStart(env, jclass, width, height, procRes, gray, cb_obj):
    initJavaClasses(env)
    global_data.config.maxImageHeight = procRes
    global_data.format = fmo.Format.GRAY if gray else fmo.Format.YUV420SP
    global_data.dims = (width, height)
    global_data.javaVM = ctypes.cast(env, ctypes.POINTER(ctypes.c_void_p)).contents.value
    global_data.stop = False
    global_data.exchange = fmo.Exchange(global_data.format, global_data.dims)
    global_data.callbackRef = Callback(env, cb_obj)

    thread = threading.Thread(target=threadImpl)
    thread.start()

def Java_cz_fmo_Lib_detectionStop(env, jclass):
    global_data.stop = True

def Java_cz_fmo_Lib_detectionFrame(env, jclass, data_yuv420sp):
    if not global_data.exchange:
        return
    data = env.GetByteArrayElements(data_yuv420sp, None)
    global_data.image.assign(global_data.format, global_data.dims, data)
    global_data.exchange.swapSend(global_data.image)
    env.ReleaseByteArrayElements(data_yuv420sp, data, 0)
