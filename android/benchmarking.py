import threading
import fmo
from java_classes import Callback
from env import Env
import ctypes

class Global:
    def __init__(self):
        self.callback_ref = None
        self.java_vm = None
        self.thread_env = None
        self.stop = False

global_data = Global()

def Java_cz_fmo_Lib_benchmarkingStart(env, jclass, cb_obj):
    initJavaClasses(env)
    global_data.callback_ref = Callback(env, cb_obj)
    global_data.stop = False
    global_data.java_vm = ctypes.cast(env, ctypes.POINTER(ctypes.c_void_p)).contents.value

    def thread_func():
        thread_env = Env(global_data.java_vm, "benchmarking")
        global_data.thread_env = thread_env.get()
        fmo.Registry.get().runAll(
            lambda c_str: global_data.callback_ref.log(c_str),
            lambda: global_data.stop
        )
        global_data.callback_ref.release(global_data.thread_env)

    thread = threading.Thread(target=thread_func)
    thread.start()

def Java_cz_fmo_Lib_benchmarkingStop(env, jclass):
    global_data.stop = True
