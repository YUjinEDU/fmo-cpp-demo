import ctypes
from java_classes import Callback, Detection, TriangleStripBuffers, FontBuffers
from env import Env
import fmo

def Java_cz_fmo_Lib_detectionStart(env, jclass, width, height, procRes, gray, cb_obj):
    initJavaClasses(env)
    global_data.config.maxImageHeight = procRes
    global_data.format = fmo.Format.GRAY if gray else fmo.Format.YUV420SP
    global_data.dims = (width, height)
    global_data.java_vm = ctypes.cast(env, ctypes.POINTER(ctypes.c_void_p)).contents.value
    global_data.stop = False
    global_data.exchange = fmo.Exchange(global_data.format, global_data.dims)
    global_data.callback_ref = Callback(env, cb_obj)

    def thread_func():
        thread_env = Env(global_data.java_vm, "Lib")
        global_data.thread_env = thread_env.get()
        frame_stats = fmo.FrameStats()
        frame_stats.reset(30)
        section_stats = fmo.SectionStats()
        input_image = fmo.Image(global_data.format, global_data.dims)
        output = fmo.Algorithm.Output()
        explorer = fmo.Algorithm.make(global_data.config, global_data.format, global_data.dims)
        callback = global_data.callback_ref.get(global_data.thread_env)
        callback.log("Detection started")

        while not global_data.stop:
            global_data.exchange.swapReceive(input_image)
            if global_data.stop:
                break

            frame_stats.tick()
            section_stats.start()
            explorer.setInputSwap(input_image)
            explorer.getOutput(output)
            stats_updated = section_stats.stop()

            if stats_updated:
                stats = statsString(section_stats)
                callback.log(stats)

            if output.detections:
                detections = DetectionArray(global_data.thread_env, len(output.detections))
                for i, det in enumerate(output.detections):
                    detection = Detection(global_data.thread_env, det)
                    detections.set(i, detection)
                callback.onObjectsDetected(detections)

        global_data.callback_ref.release(global_data.thread_env)

    thread = threading.Thread(target=thread_func)
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

def Java_cz_fmo_Lib_generateCurve(env, jclass, det_obj, rgba, b_obj):
    initJavaClasses(env)
    DECAY_BASE = 0.33
    DECAY_RATE = 0.50

    b = TriangleStripBuffers(env, b_obj, False)
    color = TriangleStripBuffers.Color()
    env.GetFloatArrayRegion(rgba, 0, 4, ctypes.byref(color))
    decay = 1.0 - DECAY_BASE

    d = Detection(env, det_obj, False)
    d_next = d.getPredecessor()
    if d_next.isNull():
        return
    pos = d.getCenter()
    pos_next = d_next.getCenter()
    dir = fmo.NormVector(pos_next - pos)
    norm = fmo.perpendicular(dir)
    radius = d.getRadius()
    v_a = shiftPoint(pos, radius, norm)
    v_b = shiftPoint(pos, -radius, norm)
    color.a = DECAY_BASE + decay
    b.addVertex(v_a, color)
    b.addVertex(v_a, color)
    b.addVertex(v_b, color)

    while True:
        d = d_next
        d_next = d.getPredecessor()
        pos = pos_next

        if d_next.isNull():
            break

        pos_next = d_next.getCenter()
        dir = fmo.NormVector(pos_next - pos)
        norm_prev = norm
        norm = fmo.perpendicular(dir)
        norm_avg = fmo.average(norm, norm_prev)
        radius = d.getRadius()
        v_a = shiftPoint(pos, radius, norm_avg)
        v_b = shiftPoint(pos, -radius, norm_avg)
        color.a = DECAY_BASE + (decay * DECAY_RATE)
        b.addVertex(v_a, color)
        b.addVertex(v_b, color)
        decay *= DECAY_RATE

    radius = d.getRadius()
    v_a = shiftPoint(pos, radius, norm)
    v_b = shiftPoint(pos, -radius, norm)
    color.a = DECAY_BASE + (decay * DECAY_RATE)
    b.addVertex(v_a, color)
    b.addVertex(v_b, color)
    b.addVertex(v_b, color)

def Java_cz_fmo_Lib_generateString(env, jclass, str, x, y, h, rgba, b_obj):
    initJavaClasses(env)
    b = FontBuffers(env, b_obj, False)
    color = FontBuffers.Color()
    env.GetFloatArrayRegion(rgba, 0, 4, ctypes.byref(color))
    cstr = env.GetStringChars(str, None)
    str_len = env.GetStringLength(str)
    char_width = 0.639
    char_step_x = 0.8 * char_width
    w = char_width * h
    ws = char_step_x * h
    pos1 = FontBuffers.Pos()
    pos2 = FontBuffers.Pos()
    pos1.y = y - 0.5 * h
    pos2.y = y + 0.5 * h
    for i in range(str_len):
        pos1.x = x + (i * ws)
        pos2.x = pos1.x + w
        ui = cstr[i] % 16
        vi = cstr[i] // 16
        uv1 = FontBuffers.UV()
        uv2 = FontBuffers.UV()
        uv1.u = ui / 16.0
        uv2.u = uv1.u + (1 / 16.0)
        uv1.v = vi / 8.0
        uv2.v = uv1.v + (1 / 8.0)
        b.addRectangle(pos1, pos2, uv1, uv2, color)
    env.ReleaseStringChars(str, cstr)

def Java_cz_fmo_Lib_generateRectangle(env, jclass, x, y, w, h, rgba, b_obj):
    initJavaClasses(env)
    b = FontBuffers(env, b_obj, False)
    pos1 = FontBuffers.Pos(x, y)
    pos2 = FontBuffers.Pos(x + w, y + h)
    uv1 = FontBuffers.UV(0.0, 0.0)
    uv2 = FontBuffers.UV(0.0625, 0.125)
    color = FontBuffers.Color()
    env.GetFloatArrayRegion(rgba, 0, 4, ctypes.byref(color))
    b.addRectangle(pos1, pos2, uv1, uv2, color)

def initJavaClasses(env):
    global DetectionBindings, TriangleStripBuffersBindings, FontBuffersBindings
    DetectionBindings = DetectionBindings(env)
    TriangleStripBuffersBindings = TriangleStripBuffersBindings(env)
    FontBuffersBindings = FontBuffersBindings(env)

def shiftPoint(center, dist, dir):
    result = TriangleStripBuffers.Pos()
    result.x = center.x + (dist * dir.x)
    result.y = center.y + (dist * dir.y)
    return result

def statsString(stats):
    q = stats.quantilesMs()
    return f"{q.q50:.2f} / {q.q95:.1f} / {q.q99:.0f}"

class Global:
    def __init__(self):
        self.callback_ref = None
        self.java_vm = None
        self.thread_env = None
        self.stop = False
        self.exchange = None
        self.config = fmo.Algorithm.Config()
        self.format = None
        self.dims = None
        self.image = fmo.Image()

global_data = Global()
