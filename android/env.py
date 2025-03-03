import ctypes

class Env:
    def __init__(self, vm, thread_name):
        self.vm = vm
        self.ptr = ctypes.c_void_p()
        args = (ctypes.c_int, thread_name, None)
        result = self.vm.AttachCurrentThread(ctypes.byref(self.ptr), args)
        assert result == 0, "AttachCurrentThread failed"

    def __del__(self):
        self.vm.DetachCurrentThread()

    def get(self):
        return self.ptr

    def __getattr__(self, name):
        return getattr(self.ptr, name)
