from ctypes import *
import numpy as np
import imageio

class rsh_wrapper:
        def __init__(self, height, width, obj_path):
                renderlib = cdll.LoadLibrary("./rsh/rsh.so")
                # int rsh_init_gl_context(int width, int height)
                renderlib.rsh_init_gl_context.restype = c_int
                renderlib.rsh_init_gl_context.argtypes = [c_int, c_int]

                # int rsh_init_renderer(int context, const char *path)
                renderlib.rsh_init_renderer.restype = c_int
                renderlib.rsh_init_renderer.argtypes = [c_int, c_char_p]

                # int rsh_destroy_renderer(int context)
                renderlib.rsh_destroy_renderer.restype = c_int
                renderlib.rsh_destroy_renderer.argtypes = [c_int]

                # int rsh_destroy_gl_context(int context)
                renderlib.rsh_destroy_gl_context.restype = c_int
                renderlib.rsh_destroy_gl_context.argtypes = [c_int]

                # unsigned char *rsh_render(int context, float *extrinsic, float *intrinsic, size_t sze, size_t szi)
                renderlib.rsh_render.restype = c_int
                renderlib.rsh_render.argtypes = [c_int, POINTER(c_uint8), c_size_t, 
                        POINTER(c_float), POINTER(c_float), c_size_t, c_size_t]

                c = renderlib.rsh_init_gl_context(width, height)
                if c < 0:
                        raise Exception('rsh_init_gl_context fail')

                r = renderlib.rsh_init_renderer(c, obj_path.encode('utf8'))
                if r < 0:
                        raise Exception('rsh_init_renderer fail')

                self.renderlib = renderlib
                self.context = c

        def clean(self):
                self.renderlib.rsh_destroy_renderer(self.context)
                self.renderlib.rsh_destroy_gl_context(self.context)

        def render_approx(self, c2w, K, buffer):
                if not buffer.dtype == np.uint8:
                        raise Exception('need float32 buffer')
                extrinsic = (c_float*16)(*c2w.reshape(-1).tolist())
                intrinsic = (c_float*9)(*K.reshape(-1).tolist())
                r = self.renderlib.rsh_render(self.context, buffer.ctypes.data_as(POINTER(c_uint8)), 
                        buffer.size, extrinsic, intrinsic, len(extrinsic), len(intrinsic))
                if r < 0:
                        raise Exception('rsh_render fail')

