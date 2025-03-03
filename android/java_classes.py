import ctypes
from env import Env

class Object:
    def __init__(self, env, obj, dispose_of_obj):
        self.env = env
        self.obj = obj
        self.obj_delete = dispose_of_obj

    def __del__(self):
        self.clear()

    def clear(self):
        if self.obj_delete and self.obj is not None:
            self.env.DeleteLocalRef(self.obj)
            self.obj = None

class Callback(Object):
    def __init__(self, env, obj):
        super().__init__(env, obj, False)
        self.class_ = env.FindClass("cz/fmo/Lib$Callback")
        self.log_method = env.GetMethodID(self.class_, "log", "(Ljava/lang/String;)V")
        self.on_objects_detected_method = env.GetMethodID(self.class_, "onObjectsDetected", "([Lcz/fmo/Lib$Detection;)V")

    def log(self, c_str):
        string = self.env.NewStringUTF(c_str)
        self.env.CallVoidMethod(self.obj, self.log_method, string)
        self.env.DeleteLocalRef(string)

    def on_objects_detected(self, detections):
        self.env.CallVoidMethod(self.obj, self.on_objects_detected_method, detections.obj)

class Detection(Object):
    def __init__(self, env, det):
        super().__init__(env, env.NewObject(DetectionBindings.class_, DetectionBindings.init_), True)
        self.env.SetIntField(self.obj, DetectionBindings.id, det.object.id)
        self.env.SetIntField(self.obj, DetectionBindings.predecessor_id, det.predecessor.id)
        self.env.SetIntField(self.obj, DetectionBindings.center_x, det.object.center.x)
        self.env.SetIntField(self.obj, DetectionBindings.center_y, det.object.center.y)
        self.env.SetFloatField(self.obj, DetectionBindings.direction_x, det.object.direction[0])
        self.env.SetFloatField(self.obj, DetectionBindings.direction_y, det.object.direction[1])
        self.env.SetFloatField(self.obj, DetectionBindings.length, det.object.length)
        self.env.SetFloatField(self.obj, DetectionBindings.radius, det.object.radius)
        self.env.SetFloatField(self.obj, DetectionBindings.velocity, det.object.velocity)

    def get_center(self):
        x = self.env.GetIntField(self.obj, DetectionBindings.center_x)
        y = self.env.GetIntField(self.obj, DetectionBindings.center_y)
        return (x, y)

    def get_radius(self):
        return self.env.GetFloatField(self.obj, DetectionBindings.radius)

    def get_predecessor(self):
        ref = self.env.GetObjectField(self.obj, DetectionBindings.predecessor)
        return Detection(self.env, ref)

class DetectionArray(Object):
    def __init__(self, env, length):
        super().__init__(env, env.NewObjectArray(length, DetectionBindings.class_, None), True)

    def set(self, i, detection):
        self.env.SetObjectArrayElement(self.obj, i, detection.obj)

class TriangleStripBuffers(Object):
    def __init__(self, env, obj, dispose_of_obj):
        super().__init__(env, obj, dispose_of_obj)
        self.num_vertices = self.env.GetIntField(self.obj, TriangleStripBuffersBindings.num_vertices)
        buf = self.env.GetObjectField(self.obj, TriangleStripBuffersBindings.pos)
        self.pos = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(Pos))
        self.env.DeleteLocalRef(buf)
        buf = self.env.GetObjectField(self.obj, TriangleStripBuffersBindings.color)
        self.color = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(Color))
        self.env.DeleteLocalRef(buf)
        self.max_vertices = min(self.env.GetDirectBufferCapacity(buf) // 2, self.env.GetDirectBufferCapacity(buf) // 4)

    def __del__(self):
        self.env.SetIntField(self.obj, TriangleStripBuffersBindings.num_vertices, self.num_vertices)

    def add_vertex(self, pos, color):
        if self.num_vertices == self.max_vertices:
            return
        self.pos[self.num_vertices] = pos
        self.color[self.num_vertices] = color
        self.num_vertices += 1

class FontBuffers(Object):
    def __init__(self, env, obj, dispose_of_obj):
        super().__init__(env, obj, dispose_of_obj)
        self.num_characters = self.env.GetIntField(self.obj, FontBuffersBindings.num_characters)
        self.num_vertices = 4 * self.num_characters
        buf = self.env.GetObjectField(self.obj, FontBuffersBindings.pos)
        self.max_characters = self.env.GetDirectBufferCapacity(buf) // 8
        self.max_vertices = self.max_characters * 4
        self.pos = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(Pos))
        self.env.DeleteLocalRef(buf)
        buf = self.env.GetObjectField(self.obj, FontBuffersBindings.uv)
        self.uv = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(UV))
        self.env.DeleteLocalRef(buf)
        buf = self.env.GetObjectField(self.obj, FontBuffersBindings.color)
        self.color = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(Color))
        self.env.DeleteLocalRef(buf)
        buf = self.env.GetObjectField(self.obj, FontBuffersBindings.idx)
        self.idx = ctypes.cast(self.env.GetDirectBufferAddress(buf), ctypes.POINTER(Idx))
        self.env.DeleteLocalRef(buf)

    def __del__(self):
        self.env.SetIntField(self.obj, FontBuffersBindings.num_characters, self.num_characters)

    def add_rectangle(self, pos1, pos2, uv1, uv2, color):
        self.add_vertex(pos1, uv1, color)
        pos = Pos()
        uv = UV()
        pos.x = pos1.x
        pos.y = pos2.y
        uv.u = uv1.u
        uv.v = uv2.v
        self.add_vertex(pos, uv, color)
        pos.x = pos2.x
        pos.y = pos1.y
        uv.u = uv2.u
        uv.v = uv1.v
        self.add_vertex(pos, uv, color)
        self.add_vertex(pos2, uv2, color)
        self.add_character()

    def add_vertex(self, pos, uv, color):
        if self.num_vertices == self.max_vertices:
            return
        self.pos[self.num_vertices] = pos
        self.uv[self.num_vertices] = uv
        self.color[self.num_vertices] = color
        self.num_vertices += 1

    def add_character(self):
        if self.num_characters == self.max_characters:
            return
        i = self.idx[self.num_characters].i
        i[0] = self.num_vertices - 4
        i[1] = self.num_vertices - 4
        i[2] = self.num_vertices - 3
        i[3] = self.num_vertices - 2
        i[4] = self.num_vertices - 1
        i[5] = self.num_vertices - 1
        self.num_characters += 1

class Pos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

class Color(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("g", ctypes.c_float), ("b", ctypes.c_float), ("a", ctypes.c_float)]

class UV(ctypes.Structure):
    _fields_ = [("u", ctypes.c_float), ("v", ctypes.c_float)]

class Idx(ctypes.Structure):
    _fields_ = [("i", ctypes.c_int * 6)]

class DetectionBindings:
    def __init__(self, env):
        self.class_ = env.FindClass("cz/fmo/Lib$Detection")
        self.init_ = env.GetMethodID(self.class_, "<init>", "()V")
        self.id = env.GetFieldID(self.class_, "id", "I")
        self.predecessor_id = env.GetFieldID(self.class_, "predecessorId", "I")
        self.center_x = env.GetFieldID(self.class_, "centerX", "I")
        self.center_y = env.GetFieldID(self.class_, "centerY", "I")
        self.direction_x = env.GetFieldID(self.class_, "directionX", "F")
        self.direction_y = env.GetFieldID(self.class_, "directionY", "F")
        self.length = env.GetFieldID(self.class_, "length", "F")
        self.radius = env.GetFieldID(self.class_, "radius", "F")
        self.velocity = env.GetFieldID(self.class_, "velocity", "F")
        self.predecessor = env.GetFieldID(self.class_, "predecessor", "Lcz/fmo/Lib$Detection;")

class TriangleStripBuffersBindings:
    def __init__(self, env):
        self.class_ = env.FindClass("cz/fmo/graphics/TriangleStripRenderer$Buffers")
        self.pos_mat = env.GetFieldID(self.class_, "posMat", "[F")
        self.pos = env.GetFieldID(self.class_, "pos", "Ljava/nio/FloatBuffer;")
        self.color = env.GetFieldID(self.class_, "color", "Ljava/nio/FloatBuffer;")
        self.num_vertices = env.GetFieldID(self.class_, "numVertices", "I")

class FontBuffersBindings:
    def __init__(self, env):
        self.class_ = env.FindClass("cz/fmo/graphics/FontRenderer$Buffers")
        self.pos_mat = env.GetFieldID(self.class_, "posMat", "[F")
        self.pos = env.GetFieldID(self.class_, "pos", "Ljava/nio/FloatBuffer;")
        self.uv = env.GetFieldID(self.class_, "uv", "Ljava/nio/FloatBuffer;")
        self.color = env.GetFieldID(self.class_, "color", "Ljava/nio/FloatBuffer;")
        self.idx = env.GetFieldID(self.class_, "idx", "Ljava/nio/IntBuffer;")
        self.num_characters = env.GetFieldID(self.class_, "numCharacters", "I")

def init_java_classes(env):
    global DetectionBindings, TriangleStripBuffersBindings, FontBuffersBindings
    DetectionBindings = DetectionBindings(env)
    TriangleStripBuffersBindings = TriangleStripBuffersBindings(env)
    FontBuffersBindings = FontBuffersBindings(env)
