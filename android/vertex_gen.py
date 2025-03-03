import ctypes
from java_classes import Detection, TriangleStripBuffers, FontBuffers
from env import Env
import fmo

def shiftPoint(center, dist, dir):
    result = TriangleStripBuffers.Pos()
    result.x = center.x + (dist * dir.x)
    result.y = center.y + (dist * dir.y)
    return result

def Java_cz_fmo_Lib_generateCurve(env, jclass, detObj, rgba, bObj):
    initJavaClasses(env)
    DECAY_BASE = 0.33
    DECAY_RATE = 0.50

    b = TriangleStripBuffers(env, bObj, False)
    color = TriangleStripBuffers.Color()
    env.GetFloatArrayRegion(rgba, 0, 4, ctypes.byref(color))
    decay = 1.0 - DECAY_BASE

    d = Detection(env, detObj, False)
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

def Java_cz_fmo_Lib_generateString(env, jclass, str, x, y, h, rgba, bObj):
    initJavaClasses(env)
    b = FontBuffers(env, bObj, False)
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

def Java_cz_fmo_Lib_generateRectangle(env, jclass, x, y, w, h, rgba, bObj):
    initJavaClasses(env)
    b = FontBuffers(env, bObj, False)
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
